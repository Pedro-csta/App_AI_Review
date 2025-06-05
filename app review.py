import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import google.generativeai as genai
from google_play_scraper import reviews, Sort, app as gp_app
from collections import Counter
from google.generativeai.types import HarmCategory, HarmProbability, GenerationConfig

# --- Configuração da Página Streamlit ---
st.set_page_config(layout="wide", page_title="Análise de Concorrência de Apps")

# --- Configuração da API Key do Gemini e Constantes ---
gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
model = None
generation_config_json = GenerationConfig(response_mime_type="application/json")

if not gemini_api_key:
    st.error("ERRO CRÍTICO: GOOGLE_API_KEY não encontrada. Configure-a nos 'Secrets' do Streamlit.")
    st.stop()
else:
    try:
        genai.configure(api_key=gemini_api_key)
        safety_settings_permissive = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmProbability.BLOCK_ONLY_HIGH},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmProbability.BLOCK_ONLY_HIGH},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmProbability.BLOCK_ONLY_HIGH},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmProbability.BLOCK_ONLY_HIGH},
        ]
        # TENTATIVA 1: Inicializar sem safety_settings personalizados para diagnóstico (se a linha abaixo falhar)
        # model = genai.GenerativeModel('gemini-1.5-flash-latest') 
        # Se a linha acima funcionar, pode tentar reabilitar com safety_settings:
        model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings_permissive)

    except Exception as e:
        st.error(f"Falha ao configurar Gemini. Tipo do Erro: {type(e)}. Detalhes: {repr(e)}")
        st.stop()

MAX_REVIEWS_TO_PROCESS = 500
APP_PLACEHOLDER_URL = "https://play.google.com/store/apps/details?id="
COMMON_NAME_INSTRUCTION = "\n\nINSTRUÇÃO IMPORTANTE SOBRE NOMES: Em toda a sua resposta textual, ao mencionar um aplicativo específico, SEMPRE use o nome completo do aplicativo exatamente como fornecido nos dados de entrada (ex: 'Meu App: Nome App XYZ', 'Concorrente 1: Nome App ABC'). NÃO generalize para 'o primeiro app' ou apenas o papel como 'Concorrente 1' em sua análise escrita."

# --- Funções Auxiliares e de Extração de Dados ---
@st.cache_data(show_spinner=False)
def get_app_id_from_url(app_url):
    if not app_url or not app_url.startswith(APP_PLACEHOLDER_URL): return None
    try: return app_url.split('id=')[1].split('&')[0]
    except IndexError: return None

def parse_short_name_from_id(app_id_str):
    if not app_id_str: return "id_desconhecido"
    parts = app_id_str.split('.')
    try:
        if parts[0].lower() == 'com' and len(parts) > 1:
            if len(parts) == 3 and parts[1].lower() == parts[2].lower(): return parts[1]
            return ".".join(parts[1:])
        elif parts[0].lower() == 'br' and len(parts) > 1 and parts[1].lower() == 'com' and len(parts) > 2:
            if len(parts) == 4 and parts[2].lower() == parts[3].lower(): return parts[2]
            return ".".join(parts[2:])
        elif len(parts) >= 2:
            if len(parts[-2]) <= 3 and len(parts) > 1 and len(parts[-1]) > 3 : return parts[-1] 
            return ".".join(parts[-2:])
        return app_id_str
    except Exception: return app_id_str

# Removido @st.cache_data de fetch_play_store_data para depurar problema de dados iguais.
# Reative se o problema de dados iguais for resolvido e o cache for benéfico.
# @st.cache_data(show_spinner="Buscando reviews e nome do app...")
def fetch_play_store_data(_app_id, lang='pt', country='br', count=MAX_REVIEWS_TO_PROCESS):
    if not _app_id: return [], [], _app_id, _app_id
    official_title = _app_id 
    short_name_from_id = parse_short_name_from_id(_app_id)
    try:
        app_details = gp_app(_app_id, lang=lang, country=country)
        official_title = app_details.get('title', _app_id) 
    except Exception as e:
        st.warning(f"Não buscou título oficial para {_app_id}. Erro: {e}")
    review_texts, review_scores = [], []
    try:
        app_reviews_data, _ = reviews(_app_id,lang=lang,country=country,sort=Sort.NEWEST,count=count,filter_score_with=None)
        for r_item in app_reviews_data:
            if r_item['content']: review_texts.append(r_item['content']); review_scores.append(r_item['score'])
        if not review_texts: st.info(f"Nenhum review para: {official_title} (ID: {_app_id}, Curto: {short_name_from_id}).")
        return review_texts, review_scores, official_title, short_name_from_id
    except Exception as e: st.error(f"Erro ao buscar reviews para {official_title} ({_app_id}): {e}"); return [],[],official_title, short_name_from_id

# --- Funções de Análise com Gemini ---
@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Analisando sentimento e temas...")
def analyze_single_app_reviews(gemini_model_instance, reviews_text, app_name=""):
    err_payload = {"app_name":app_name,"sentiment_summary":{"error":100.0, "message":"Falha na análise de sent/temas"},"top_topics":[]}
    if not reviews_text.strip(): return {"app_name":app_name,"sentiment_summary":{"no_reviews":100.0},"top_topics":[]}
    reviews_sample = str(reviews_text)[:10000]; num_reviews = len(reviews_sample.splitlines()); ctx = f"do app '{app_name}'" if app_name else "de um app"
    p = f"""Analise os reviews de '{app_name}'. {num_reviews} reviews de amostra.
Sua resposta DEVE SER APENAS um objeto JSON válido, começando com {{ e terminando com }}. Não inclua NENHUM texto antes ou depois.
JSON: {{ "app_name":"{app_name}", "sentiment_summary":{{...}}, "top_topics":[...] }}
sentiment_summary: % Pos,Neu,Neg,SemSent. Soma 100%.
top_topics: 5-7 temas (nome, menções P,N,Ng).
REVIEWS: {reviews_sample}
{COMMON_NAME_INSTRUCTION}"""
    response_text_debug = ""
    try:
        response = gemini_model_instance.generate_content(p, generation_config=generation_config_json)
        if not response.candidates:
             st.warning(f"Resposta API vazia (sent/temas) '{app_name}'. Feedback: {response.prompt_feedback}")
             return {**err_payload, "sentiment_summary": {**err_payload["sentiment_summary"], "message": f"API Vazia. Feedback: {response.prompt_feedback}"}}
        response_text_debug = response.text
        if not response_text_debug:
            st.warning(f"Texto da resposta API vazio (sent/temas) '{app_name}'.")
            return {**err_payload, "sentiment_summary": {**err_payload["sentiment_summary"], "message": "Texto API vazio"}}
        d = json.loads(response_text_debug)
        s = d.get("sentiment_summary",{})
        s_no_na={k:v for k,v in s.items() if k!='no_sentiment_detected'}
        s['no_sentiment_detected']=round(max(0,100.0-sum(s_no_na.values())),2)
        ts=sum(s.values());s={k:(v/ts*100 if ts>0 else 0) for k,v in s.items()} if abs(ts-100.0)>0.01 and ts!=0 else s
        d["sentiment_summary"]={k:round(v,2) for k,v in s.items()}
        final_sum=sum(d["sentiment_summary"].values())
        if abs(final_sum-100.0)>0.01 and 'no_sentiment_detected' in d["sentiment_summary"]:
            d["sentiment_summary"]['no_sentiment_detected']+= (100.0-final_sum)
            d["sentiment_summary"]['no_sentiment_detected']=round(max(0,d["sentiment_summary"]['no_sentiment_detected']),2)
        return d
    except json.JSONDecodeError as e: st.error(f"JSONError (sent/temas) '{app_name}': {e}. Resposta (limitada):\n'{str(response_text_debug)[:200]}...'"); return err_payload
    except Exception as e: st.error(f"Erro (sent/temas) '{app_name}': {e}. Resposta (limitada):\n'{str(response_text_debug)[:200]}...'"); return err_payload

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Extraindo features...")
def extract_feature_details_from_reviews(gemini_model_instance, app_name, review_texts_str):
    err_payload = {"elogiadas":[],"problematicas":[],"desejadas_ausentes":[{"funcionalidade":"Erro Análise","descricao_usuario":"Falha"}]}
    if not review_texts_str.strip(): return err_payload
    p = f"""Reviews de '{app_name}'. JSON: 'elogiadas','problematicas','desejadas_ausentes'. Cada: 'funcionalidade','descricao_usuario'. Max 3-5/cat. REVIEWS: {str(review_texts_str)[:8000]} {COMMON_NAME_INSTRUCTION}"""
    try:
        r = gemini_model_instance.generate_content(p, generation_config=generation_config_json).text
        if not r: return err_payload
        return json.loads(r)
    except Exception as e: st.error(f"Erro (features) '{app_name}': {e}"); return err_payload

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Extraindo dor/encantamento...")
def extract_pain_delight_points_from_reviews(gemini_model_instance, app_name, review_texts_str):
    err_payload = {"pontos_dor":[{"tipo":"Erro","descricao":"Falha"}],"fatores_encantamento":[]}
    if not review_texts_str.strip(): return err_payload
    p = f"""Reviews de '{app_name}'. JSON: 'pontos_dor'(3-5 frustrações),'fatores_encantamento'(3-5 satisfações). Cada:'tipo','descricao','ex_review'. REVIEWS: {str(review_texts_str)[:8000]} {COMMON_NAME_INSTRUCTION}"""
    try:
        r = gemini_model_instance.generate_content(p, generation_config=generation_config_json).text
        if not r: return err_payload
        return json.loads(r)
    except Exception as e: st.error(f"Erro (dor/enc.) '{app_name}': {e}"); return err_payload

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Sintetizando GAPs de features...")
def synthesize_feature_gap_analysis(gemini_model_instance, all_apps_feature_details, my_app_name):
    p = f"""Analista de produto. Features (incl. '{my_app_name}'): {json.dumps(all_apps_feature_details,ensure_ascii=False)}. GAPs/oportunidades (Markdown):
1. Forças '{my_app_name}'. 2. Fraquezas/GAPs '{my_app_name}'. 3. Oportunidades '{my_app_name}' (2-3). 4. Ameaças Concorrentes.{COMMON_NAME_INSTRUCTION}"""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (GAPs): {e}"); return f"Erro: {e}"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Sintetizando dor/encantamento...")
def synthesize_pain_delight_comparison(gemini_model_instance, all_apps_pain_delight_details, my_app_name):
    p = f"""UX Expert. Dor/encantamento (incl. '{my_app_name}'): {json.dumps(all_apps_pain_delight_details,ensure_ascii=False)}. Análise comparativa (Markdown):
1. Dores Comuns. '{my_app_name}' sofre? 2. Dores Críticas '{my_app_name}'. 3. Encantamentos Diferenciais '{my_app_name}'. 4. Inspirações Concorrentes. 5. Recomendações '{my_app_name}'.{COMMON_NAME_INSTRUCTION}"""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (comp. dor/enc.): {e}"); return f"Erro: {e}"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sobre notas...")
def analyze_ratings_insights(gemini_model_instance, app_name, scores_distribution_str):
    p = f"App '{app_name}', dist. notas:\n{scores_distribution_str}\nParágrafo (máx 60 palavras) análise profissional. Use '{app_name}'.{COMMON_NAME_INSTRUCTION}"
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (ratings) '{app_name}': {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sobre temas...")
def analyze_topics_insights(gemini_model_instance, app_name, topics_data_json_str):
    p = f"App '{app_name}', temas:\n{topics_data_json_str}\nParágrafo (máx 70 palavras) 1-2 temas +/- e implicações. Use '{app_name}'.{COMMON_NAME_INSTRUCTION}"
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (topics) '{app_name}': {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sentimento comparativo...")
def analyze_comparative_sentiment_insights(gemini_model_instance, sentiment_comparison_str):
    p = f"Sentimento comparativo:\n{sentiment_comparison_str}\nParágrafo (máx 70 palavras) comparando perfis e implicações.{COMMON_NAME_INSTRUCTION}"
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (sent. comp.): {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando análise qualitativa estruturada...")
def generate_competitive_qualitative_analysis(gemini_model_instance, all_apps_sentiment_topic_analyses, my_app_name):
    err_payload = {"analises_individuais":[{"app_name":my_app_name,"pontos_fortes":["Erro"],"pontos_fracos":[]}],"oportunidades_mercado":[],"ameacas_desafios_mercado":[],"tendencias_emergentes":[],"sugestoes_posicionamento_meu_app":[]}
    inp = "\n\n".join([f"App: {a['app_name']}\n- Sentimento: {json.dumps(a['sentiment_summary'])}\n- Temas: {json.dumps(a['top_topics'])}" for a in all_apps_sentiment_topic_analyses if "error" not in a.get("sentiment_summary",{})])
    if not inp: return err_payload
    p = f"""Analista de mercado. Dados de sentimento/temas (incl. '{my_app_name}'): {inp}
{COMMON_NAME_INSTRUCTION}
Retorne JSON com chaves: "analises_individuais":[{{"app_name","pontos_fortes":[],"pontos_fracos":[]}}],"oportunidades_mercado":[],"ameacas_desafios_mercado":[],"tendencias_emergentes":[],"sugestoes_posicionamento_meu_app":[] (para '{my_app_name}')."""
    try:
        r = gemini_model_instance.generate_content(p, generation_config=generation_config_json).text
        if not r: return err_payload
        return json.loads(r)
    except Exception as e: st.error(f"Erro (análise qual.): {e}"); return err_payload

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando Matriz SWOT...")
def generate_swot_analysis(gemini_model_instance, my_app_name, my_app_strengths, my_app_weaknesses, market_opportunities, market_threats):
    s,w,o,t = ["- "+"\n- ".join(l) if l else "Nenhuma informação específica identificada." for l in [my_app_strengths,my_app_weaknesses,market_opportunities,market_threats]]
    p = f"""Crie Matriz SWOT (Markdown) para '{my_app_name}'.
Forças:\n{s}\nFraquezas:\n{w}\nOportunidades:\n{o}\nAmeaças:\n{t}
Formato: ## Matriz SWOT para {my_app_name}\n### Forças\n- ...
{COMMON_NAME_INSTRUCTION}"""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (SWOT): {e}"); return "Erro SWOT."

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando perfil do público...")
def synthesize_audience_profile(gemini_model_instance, all_apps_collected_data_str, my_app_name_context):
    p = f"""Com base nos dados de reviews de '{my_app_name_context}' e concorrentes, crie perfil do público da categoria (Markdown).
{COMMON_NAME_INSTRUCTION}
* Visão Geral Público: Quem são? Objetivos?
* Dores Comuns?
* O que Valorizam?
* Objeções/Preocupações?
* Perfil de Uso?
* Outras Infos Qualitativas?
Dados: {all_apps_collected_data_str}"""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (perfil público): {e}"); return "Erro perfil público."

# --- Funções de Visualização ---
def plot_comparative_sentiment_chart(all_apps_sentiment_data):
    if not all_apps_sentiment_data: st.info("Nenhum dado de sentimento para plotar."); return
    plot_data = {}; categories = ['positive','neutral','negative','no_sentiment_detected']
    display_labels = {'positive':'Positivo','neutral':'Neutro','negative':'Negativo','no_sentiment_detected':'Não Detectado'}
    app_names = []; valid_data_found = False
    for d_item in all_apps_sentiment_data:
        if d_item and "app_name" in d_item and "sentiment_summary" in d_item and "error" not in d_item["sentiment_summary"]:
            valid_data_found=True; app_n=d_item['app_name'];app_names.append(app_n)
            for cat in categories:
                if cat not in plot_data: plot_data[cat]=[]
                plot_data[cat].append(d_item['sentiment_summary'].get(cat,0))
    if not valid_data_found: st.info("Nenhum app com dados de sentimento válidos."); return
    df=pd.DataFrame(plot_data,index=app_names).rename(columns=display_labels)
    colors={'Positivo':'#4CAF50','Neutro':'#FFC107','Negativo':'#F44336','Não Detectado':'#9E9E9E'}
    df_filtered=df[[c for c in df.columns if c in df and df[c].sum()>0 and c in colors]]
    if df_filtered.empty: st.info("Dados de sentimento zerados."); return
    
    fig,ax=plt.subplots(figsize=(max(10,len(app_names)*2.0),6))
    df_filtered.plot(kind='bar',ax=ax,color=[colors[c] for c in df_filtered.columns])
    ax.set_title('Análise Comparativa de Sentimentos (%)',fontsize=15)
    ax.set_ylabel('Porcentagem de Reviews (%)');ax.set_xlabel('Aplicativos')
    ax.legend(title='Sentimento')
    # CORREÇÃO DO ValueError: Usar plt.setp para manipular rótulos do eixo X
    if ax.get_xticklabels():
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=9)
    plt.tight_layout();st.pyplot(fig)

def plot_topics_chart_for_app(app_analysis_data):
    app_n=app_analysis_data.get('app_name','App Desconhecido');topics_data=app_analysis_data.get('top_topics',[])
    if not topics_data: st.info(f"Nenhum tema para {app_n}."); return
    parsed_topics=[]
    for t in topics_data:
        if "name" in t and "mentions" in t: parsed_topics.append({'name':t['name'],'positive':t['mentions'].get('positive',0),'neutral':t['mentions'].get('neutral',0),'negative':t['mentions'].get('negative',0)})
    if not parsed_topics: st.info(f"Temas malformatados para {app_n}."); return
    df_topics=pd.DataFrame(parsed_topics);df_topics['Total']=df_topics['positive']+df_topics['neutral']+df_topics['negative']
    df_topics=df_topics[df_topics['Total']>0]
    if df_topics.empty: st.info(f"Nenhum tema com menções para {app_n}."); return
    df_topics=df_topics.sort_values('Total',ascending=True).tail(10)
    fig,ax=plt.subplots(figsize=(8,max(4,len(df_topics)*0.55)))
    bar_colors_map={'positive':'#4CAF50','neutral':'#FFC107','negative':'#F44336'}
    df_plot=df_topics[['name','positive','neutral','negative']].set_index('name')
    df_plot.plot(kind='barh',stacked=True,color=[bar_colors_map['positive'],bar_colors_map['neutral'],bar_colors_map['negative']],ax=ax)
    ax.set_title(f'Principais Temas por Sentimento - {app_n}',fontsize=14)
    ax.set_xlabel('Nº de Menções');ax.set_ylabel('Tema')
    ax.tick_params(axis='x',colors='black');ax.tick_params(axis='y',colors='black',labelsize=9)
    ax.legend(['Positivo','Neutro','Negativo'],loc='lower right',frameon=False)
    plt.tight_layout();st.pyplot(fig)

def plot_ratings_distribution_chart(app_name,review_scores):
    if not review_scores: st.info(f"Nenhuma nota para {app_name}."); return "",None
    score_counts=Counter(review_scores);ratings_df=pd.DataFrame({'Estrelas':range(1,6)})
    counts_df=pd.DataFrame(list(score_counts.items()),columns=['Estrelas','Contagem']).sort_values(by='Estrelas')
    ratings_df=pd.merge(ratings_df,counts_df,on='Estrelas',how='left').fillna(0)
    ratings_df['Contagem']=ratings_df['Contagem'].astype(int)
    max_count_for_plot = max(1, ratings_df['Contagem'].max()) 
    fig,ax=plt.subplots(figsize=(7,5))
    bars=ax.bar(ratings_df['Estrelas'],ratings_df['Contagem'],color=['#D32F2F','#F57C00','#FFB300','#7CB342','#43A047'])
    ax.set_title(f'Distribuição de Notas - {app_name}',fontsize=14)
    ax.set_xlabel('Nota (Estrelas)');ax.set_ylabel('Nº de Reviews');ax.set_xticks(range(1,6))
    for bar in bars:
        yval=bar.get_height()
        if yval>0: plt.text(bar.get_x()+bar.get_width()/2.0, yval+0.05*max_count_for_plot, int(yval), ha='center',va='bottom')
    plt.tight_layout();st.pyplot(fig)
    total_r_score=sum(ratings_df['Contagem'])
    if total_r_score==0: return "Nenhuma nota válida.",None
    dist_str=f"Total reviews c/ notas: {total_r_score}.\n" + "\n".join([f"- {r['Estrelas']}*: {r['Contagem']} ({r['Contagem']*100/total_r_score:.1f}%)" for i,r in ratings_df.iterrows()])
    return dist_str,ratings_df

# --- Interface Streamlit e Lógica Principal ---
st.title("🤖 Ferramenta Avançada de Análise de Concorrência de Apps")
st.markdown(f"Insira URLs da Play Store. Máx **{MAX_REVIEWS_TO_PROCESS} reviews**/app.")

with st.sidebar:
    st.header("🔗 URLs dos Aplicativos")
    my_app_url_st = st.text_input("URL Seu App (Foco):", key="my_app_url")
    competitor_urls_st = [st.text_input(f"URL Concorrente {i+1}:",key=f"comp{i}_url") for i in range(3)]
    competitor_urls_st = [url for url in competitor_urls_st if url] # Filtra strings vazias
    analyze_button = st.button("🔍 Analisar Aplicativos", type="primary", use_container_width=True)

# Inicialização do Estado da Sessão
default_session_state = {
    'analysis_complete': False, 'all_apps_processed_data': [], 'feature_gap_report': "",
    'pain_delight_report': "", 'overall_qualitative_data': {}, 'swot_report': "",
    'audience_profile_report': "", 'my_app_name_for_synthesis': ""
}
for k, v in default_session_state.items():
    if k not in st.session_state: st.session_state[k] = v

if analyze_button:
    if not my_app_url_st: st.sidebar.warning("Insira a URL do seu app.")
    elif not model: st.error("Modelo Gemini não inicializado.")
    else:
        # Resetar estado para nova análise
        for k, v_default in default_session_state.items(): st.session_state[k] = v_default
        
        urls_to_process=[{"url":my_app_url_st,"role":"Meu App"}]
        urls_to_process.extend([{"url":url,"role":f"Concorrente {i+1}"} for i,url in enumerate(competitor_urls_st)])
        
        prog_bar=st.sidebar.progress(0); prog_text=st.sidebar.empty()
        total_steps=len(urls_to_process)*4 + 6 # fetch + 3 AI/app + 6 global AI
        
        # Adicionando DEBUG INICIAL na Sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("📢 DEBUG DA COLETA (Sidebar)")
        debug_placeholder = st.sidebar.empty()
        cumulative_debug_log = "" # Para acumular logs de debug
        
        with st.spinner("Análise em progresso... pode levar minutos."):
            processed_data_list=[]
            step_counter=0 

            for i,app_info in enumerate(urls_to_process):
                app_id=get_app_id_from_url(app_info['url'])
                
                current_debug_log_for_app = f"**Processando: {app_info['role']}**\n- URL: {app_info['url']}\n- App ID Parseado: {app_id}\n"
                
                initial_short_name = parse_short_name_from_id(app_id) if app_id else "ID_Inválido"
                # Inicializa com nome baseado no ID para o caso de falha na busca do título
                app_data_current={"display_name":f"{app_info['role']}: {initial_short_name}",
                                  "review_texts_str":"","review_scores":[],
                                  "sentiment_topic_analysis":{},"feature_details":{},"pain_delight_points":{}}
                
                if app_id:
                    step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) Coleta: {app_info['role']} ({initial_short_name})")
                    
                    texts,scores,official_title,short_name_id = fetch_play_store_data(app_id) # short_name_id já vem parseado
                    current_debug_log_for_app += f"- Título Oficial Retornado: {official_title}\n- Short Name do ID Retornado: {short_name_id}\n"
                    current_debug_log_for_app += f"- Reviews Coletados: {len(texts)}, Notas Coletadas: {len(scores)}\n"
                    if texts: current_debug_log_for_app += f"- Amostra Review[0]: '{texts[0][:50]}...'\n- Amostra Notas[:3]: {scores[:3]}\n"
                    else: current_debug_log_for_app += "- Nenhum review/nota coletado.\n"
                    
                    # Lógica de Nome de Exibição (display_name) REFINADA
                    if app_info['role'] == "Meu App":
                        d_name = f"Meu App: {official_title if official_title and official_title != app_id else short_name_id}"
                        if short_name_id and short_name_id.lower() not in d_name.lower() and app_id.lower() != official_title.lower() :
                             d_name += f" [{short_name_id}]"
                    else: # Concorrentes
                        d_name = f"{app_info['role']}: {short_name_id}" # Prioriza short_name_id
                        if official_title and official_title != app_id and official_title.lower() != short_name_id.lower():
                             d_name += f" ({official_title})" # Adiciona título oficial se útil
                    
                    current_debug_log_for_app += f"- Display Name Gerado: {d_name}\n"
                    app_data_current.update({"display_name":d_name,"review_scores":scores})
                    if app_info['role']=="Meu App":st.session_state.my_app_name_for_synthesis=d_name
                    
                    if texts:
                        r_str="\n".join(texts);app_data_current["review_texts_str"]=r_str
                        step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) Sent/Temas: {d_name[:30]}...")
                        app_data_current["sentiment_topic_analysis"]=analyze_single_app_reviews(model,r_str,d_name)
                        step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) Features: {d_name[:30]}...")
                        app_data_current["feature_details"]=extract_feature_details_from_reviews(model,d_name,r_str)
                        step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) Dor/Enc: {d_name[:30]}...")
                        app_data_current["pain_delight_points"]=extract_pain_delight_points_from_reviews(model,d_name,r_str)
                    else:app_data_current["sentiment_topic_analysis"]={"app_name":d_name,"sentiment_summary":{"no_reviews":100.0},"top_topics":[]}
                else:st.warning(f"URL inválida para {app_info['role']}. Pulando.")
                
                cumulative_debug_log += current_debug_log_for_app + "---\n" # Acumula logs de debug
                debug_placeholder.markdown(cumulative_debug_log) # Exibe logs acumulados
                processed_data_list.append(app_data_current)
            st.session_state.all_apps_processed_data=processed_data_list

            if processed_data_list: # Início das Sínteses Globais
                my_app_n=st.session_state.my_app_name_for_synthesis
                if not my_app_n and processed_data_list and processed_data_list[0]: # Fallback se my_app_name não foi definido
                    my_app_n=processed_data_list[0]["display_name"];st.session_state.my_app_name_for_synthesis=my_app_n
                
                # (Resto da lógica de SÍNTESES GLOBAIS e ATUALIZAÇÃO DE ESTADO como antes)
                step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) GAPs Features...")
                feat_l=[{"app_name":d["display_name"],**d["feature_details"]} for d in processed_data_list if d.get("feature_details")]
                if feat_l and my_app_n:st.session_state.feature_gap_report=synthesize_feature_gap_analysis(model,feat_l,my_app_n)

                step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) Comp. Dor/Enc...")
                pain_l=[{"app_name":d["display_name"],**d["pain_delight_points"]} for d in processed_data_list if d.get("pain_delight_points")]
                if pain_l and my_app_n:st.session_state.pain_delight_report=synthesize_pain_delight_comparison(model,pain_l,my_app_n)
                
                step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) Análise Qualitativa Geral...")
                sent_topic_l=[d["sentiment_topic_analysis"] for d in processed_data_list if d.get("sentiment_topic_analysis") and "no_reviews" not in d["sentiment_topic_analysis"].get("sentiment_summary",{})]
                if sent_topic_l and my_app_n:
                    st.session_state.overall_qualitative_data=generate_competitive_qualitative_analysis(model,sent_topic_l,my_app_n)
                    step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) SWOT {my_app_n[:25]}...")
                    q_data=st.session_state.overall_qualitative_data
                    my_app_q_an=next((item for item in q_data.get("analises_individuais",[]) if item.get("app_name")==my_app_n),None)
                    if my_app_q_an and q_data: 
                        st.session_state.swot_report=generate_swot_analysis(model,my_app_n,my_app_q_an.get("pontos_fortes",[]),my_app_q_an.get("pontos_fracos",[]),q_data.get("oportunidades_mercado",[]),q_data.get("ameacas_desafios_mercado",[]))
                
                step_counter+=1; prog_bar.progress(min(1.0,step_counter/total_steps)); prog_text.info(f"({step_counter}/{total_steps}) Perfil Público...")
                aud_prompt_data=[{"app_name":d["display_name"],"sentiment":d["sentiment_topic_analysis"].get("sentiment_summary"),"top_topics":d["sentiment_topic_analysis"].get("top_topics"),"requested_features":[f.get('funcionalidade') for f in d.get("feature_details",{}).get("desejadas_ausentes",[])],"pain_points":[p.get('descricao') for p in d.get("pain_delight_points",{}).get("pontos_dor",[])],"delight_factors":[dl.get('descricao') for dl in d.get("pain_delight_points",{}).get("fatores_encantamento",[])]} for d in processed_data_list]
                if aud_prompt_data and my_app_n:st.session_state.audience_profile_report=synthesize_audience_profile(model,json.dumps(aud_prompt_data,ensure_ascii=False,indent=2),my_app_n)
            
            st.session_state.analysis_complete=True
            prog_text.success("Análise Concluída!");prog_bar.progress(1.0)
            # debug_placeholder.empty() # Mantém os logs de debug visíveis na sidebar após a execução

# --- Seção de Display dos Resultados (como antes, com as 7 abas) ---
if st.session_state.analysis_complete and st.session_state.all_apps_processed_data:
    st.header("🏁 Resultados da Análise Competitiva")
    tab_names = ["📊 Sent. Geral","📱 Apps Individuais","🧩 GAPs Features","❤️ Dor vs. Enc.","♟️ SWOT","👤 Perfil Público","💡 Qualitativa Estratégica"]
    tabs=st.tabs(tab_names)
    my_app_n_disp = st.session_state.my_app_name_for_synthesis

    with tabs[0]: # Sentimento Geral
        # ... (código da aba como antes) ...
        st.subheader(tab_names[0])
        valid_sent=[d["sentiment_topic_analysis"] for d in st.session_state.all_apps_processed_data if d.get("sentiment_topic_analysis") and "no_reviews" not in d["sentiment_topic_analysis"].get("sentiment_summary",{}) and "error" not in d["sentiment_topic_analysis"].get("sentiment_summary",{})]
        if valid_sent:
            plot_comparative_sentiment_chart(valid_sent)
            sent_comp_str="\n".join([f"- App {s['app_name']}: Pos={s['sentiment_summary'].get('positive',0)}%,Neu={s['sentiment_summary'].get('neutral',0)}%,Neg={s['sentiment_summary'].get('negative',0)}%" for s in valid_sent])
            if sent_comp_str and model: st.markdown(f"**Análise IA:**\n{analyze_comparative_sentiment_insights(model,sent_comp_str)}")
        else: st.info("Dados insuficientes.")

    with tabs[1]: # Apps Individuais
        # ... (código da aba como antes) ...
        st.subheader(tab_names[1])
        for app_d in st.session_state.all_apps_processed_data:
            d_name=app_d["display_name"]
            is_my_app = my_app_n_disp == d_name
            with st.expander(f"Detalhes: {d_name}",expanded=is_my_app):
                st.markdown(f"**Distribuição de Notas**")
                if app_d["review_scores"]:
                    r_str,_=plot_ratings_distribution_chart(d_name,app_d["review_scores"])
                    if r_str and "Nenhuma nota" not in r_str and model:st.markdown(f"**Análise IA:**\n{analyze_ratings_insights(model,d_name,r_str)}")
                else:st.caption(f"Sem notas.")
                st.markdown("---")
                st.markdown(f"**Principais Temas**")
                s_t_an=app_d["sentiment_topic_analysis"]
                if s_t_an and "no_reviews" not in s_t_an.get("sentiment_summary",{}) and "error" not in s_t_an.get("sentiment_summary",{}) and s_t_an.get("top_topics"):
                    plot_topics_chart_for_app(s_t_an)
                    topics_json=json.dumps(s_t_an.get("top_topics",[]),ensure_ascii=False,indent=2)
                    if model:st.markdown(f"**Análise IA:**\n{analyze_topics_insights(model,d_name,topics_json)}")
                    with st.popover("JSON temas"):st.json(s_t_an.get("top_topics",[]))
                else:st.caption(f"Sem temas/reviews.")
                st.markdown("---")
                st.markdown(f"**Detalhes de Funcionalidades (Extraído)**")
                if app_d.get("feature_details") and any(app_d["feature_details"].values()):st.json(app_d["feature_details"])
                else:st.caption(f"Não extraído.")
                st.markdown("---")
                st.markdown(f"**Pontos de Dor e Encantamento (Extraído)**")
                if app_d.get("pain_delight_points") and any(app_d["pain_delight_points"].values()):st.json(app_d["pain_delight_points"])
                else:st.caption(f"Não extraído.")
    
    with tabs[2]:st.subheader(tab_names[2]);st.markdown(st.session_state.feature_gap_report if st.session_state.feature_gap_report else "Não gerado.")
    with tabs[3]:st.subheader(tab_names[3]);st.markdown(st.session_state.pain_delight_report if st.session_state.pain_delight_report else "Não gerado.")
    with tabs[4]:st.subheader(f"{tab_names[4]} ({my_app_n_disp if my_app_n_disp else 'Meu App'})");st.markdown(st.session_state.swot_report if st.session_state.swot_report else "Não gerado.")
    with tabs[5]:st.subheader(tab_names[5]);st.markdown(st.session_state.audience_profile_report if st.session_state.audience_profile_report else "Não gerado.")
    with tabs[6]:
        # ... (código da aba como antes) ...
        st.subheader(tab_names[6])
        q_data=st.session_state.get("overall_qualitative_data",{})
        if q_data and q_data.get("analises_individuais"):
            st.markdown("#### Análises Individuais (Resumo Estratégico)")
            for app_an in q_data.get("analises_individuais",[]):
                st.markdown(f"**App: {app_an.get('app_name')}**");st.markdown(f"* Fortes: {'; '.join(app_an.get('pontos_fortes',['N/A']))}");st.markdown(f"* Fracos: {'; '.join(app_an.get('pontos_fracos',['N/A']))}");st.markdown("---")
            st.markdown("#### Oportunidades de Mercado");st.markdown("\n".join([f"- {op}" for op in q_data.get("oportunidades_mercado",["N/A"])]))
            st.markdown("#### Ameaças e Desafios");st.markdown("\n".join([f"- {am}" for am in q_data.get("ameacas_desafios_mercado",["N/A"])]))
            st.markdown("#### Tendências Emergentes");st.markdown("\n".join([f"- {te}" for te in q_data.get("tendencias_emergentes",["N/A"])]))
            my_app_n_s=my_app_n_disp if my_app_n_disp else "Seu App"
            st.markdown(f"#### Sugestões de Posicionamento para {my_app_n_s}")
            st.markdown("\n".join([f"- {sp}" for sp in q_data.get("sugestoes_posicionamento_meu_app",["N/A"])]))
        else:st.info("Análise qualitativa geral não gerada.")


elif not analyze_button:
    st.info("⬅️ Insira URLs na barra lateral e clique em 'Analisar Aplicativos'.")

st.sidebar.markdown("---");st.sidebar.caption("✨ IA Avançada ✨")
