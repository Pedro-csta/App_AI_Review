import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import google.generativeai as genai
from google_play_scraper import reviews, Sort, app as gp_app
from collections import Counter
from google.generativeai.types import HarmCategory, HarmProbability, GenerationConfig # Adicionado GenerationConfig

# --- Configuração da Página Streamlit ---
st.set_page_config(layout="wide", page_title="Análise de Concorrência de Apps")

# --- Configuração da API Key do Gemini e Constantes ---
gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
model = None
generation_config_json = GenerationConfig(response_mime_type="application/json") # Config para forçar JSON

if not gemini_api_key:
    st.error("ERRO CRÍTICO: GOOGLE_API_KEY não encontrada. Configure-a nos 'Secrets' do Streamlit.")
    st.stop()
else:
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Falha ao configurar Gemini: {e}"); st.stop()

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
    """Extrai um nome curto do ID do app, como 'supermei' de 'com.supermei.supermei'."""
    if not app_id_str: return "id_desconhecido"
    parts = app_id_str.split('.')
    if len(parts) > 1:
        # Se começar com 'com' ou 'br', pega o penúltimo ou último se for relevante
        if parts[0] == 'com' and len(parts) > 1:
            potential_name = parts[-1] # Pega o último por padrão
            if len(parts) > 2 and parts[-2] != parts[0] and len(parts[-2]) > 2 : # ex: com.nomeempresa.nomeapp
                potential_name = parts[-2] + "." + parts[-1] if len(parts[-1]) < 4 else parts[-1] # tenta ser um pouco mais esperto
            elif len(parts) == 2: # ex: com.nomeapp
                 potential_name = parts[1]
            return potential_name
        # para br.com.empresa.app ou org.empresa.app etc.
        return parts[-1] # Padrão: último segmento
    return app_id_str # Retorna o ID inteiro se não for um formato comum

@st.cache_data(show_spinner="Buscando reviews e nome do app...")
def fetch_play_store_data(_app_id, lang='pt', country='br', count=MAX_REVIEWS_TO_PROCESS):
    if not _app_id: return [], [], _app_id, _app_id # texts, scores, official_title, short_name
    official_title = _app_id
    short_name_from_id = parse_short_name_from_id(_app_id)
    try:
        app_details = gp_app(_app_id, lang=lang, country=country)
        official_title = app_details.get('title', _app_id)
    except Exception: pass
    review_texts, review_scores = [], []
    try:
        app_reviews_data, _ = reviews(_app_id,lang=lang,country=country,sort=Sort.NEWEST,count=count,filter_score_with=None)
        for r_item in app_reviews_data:
            if r_item['content']: review_texts.append(r_item['content']); review_scores.append(r_item['score'])
        if not review_texts: st.info(f"Nenhum review para: {official_title} ({short_name_from_id}).")
        return review_texts, review_scores, official_title, short_name_from_id
    except Exception as e: st.error(f"Erro ao buscar reviews para {official_title} ({_app_id}): {e}"); return [],[],official_title, short_name_from_id

# --- Funções de Análise com Gemini (Aplicando GenerationConfig para JSON) ---
@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Analisando sentimento e temas...")
def analyze_single_app_reviews(gemini_model_instance, reviews_text, app_name=""):
    # (Função modificada para usar generation_config_json e melhor tratamento de erro)
    err_payload = {"app_name":app_name,"sentiment_summary":{"error":100.0, "message":"Falha na análise"},"top_topics":[]}
    if not reviews_text.strip(): return {"app_name":app_name,"sentiment_summary":{"no_reviews":100.0},"top_topics":[]}
    
    # Reduzido ainda mais para robustez, e garantindo que é uma string
    reviews_sample = str(reviews_text)[:10000]

    p = f"""Analise os reviews de '{app_name}'. {len(reviews_sample.splitlines())} reviews de amostra.
Sua resposta DEVE SER APENAS um objeto JSON válido.
JSON: {{ "app_name":"{app_name}", "sentiment_summary":{{...}}, "top_topics":[...] }}
sentiment_summary: % Pos,Neu,Neg,SemSent. Soma 100%.
top_topics: 5-7 temas (nome, menções P,N,Ng).
REVIEWS: {reviews_sample}
{COMMON_NAME_INSTRUCTION}"""
    response_text_debug = ""
    try:
        response = gemini_model_instance.generate_content(p, generation_config=generation_config_json)
        if not response.candidates:
             st.error(f"Resposta API vazia (sent/temas) '{app_name}'. Feedback: {response.prompt_feedback}")
             return {**err_payload, "sentiment_summary": {**err_payload["sentiment_summary"], "message": f"API Vazia. Feedback: {response.prompt_feedback}"}}
        # (Checagens de prompt_feedback e safety_ratings como na versão anterior, se necessário)
        response_text_debug = response.text.strip()
        if not response_text_debug:
            st.error(f"Texto da resposta API vazio (sent/temas) '{app_name}'.")
            return {**err_payload, "sentiment_summary": {**err_payload["sentiment_summary"], "message": "Texto API vazio"}}
        
        d = json.loads(response_text_debug) # Não precisa mais de limpeza de ```json
        s = d.get("sentiment_summary",{}) # (Normalização do sentimento como antes)
        s_no_na={k:v for k,v in s.items() if k!='no_sentiment_detected'}
        s['no_sentiment_detected']=round(max(0,100.0-sum(s_no_na.values())),2)
        ts=sum(s.values());s={k:(v/ts*100 if ts>0 else 0) for k,v in s.items()} if abs(ts-100.0)>0.01 and ts!=0 else s
        d["sentiment_summary"]={k:round(v,2) for k,v in s.items()}
        final_sum=sum(d["sentiment_summary"].values())
        if abs(final_sum-100.0)>0.01 and 'no_sentiment_detected' in d["sentiment_summary"]:
            d["sentiment_summary"]['no_sentiment_detected']+= (100.0-final_sum)
            d["sentiment_summary"]['no_sentiment_detected']=round(max(0,d["sentiment_summary"]['no_sentiment_detected']),2)
        return d
    except json.JSONDecodeError as e: st.error(f"JSONDecodeError (sent/temas) '{app_name}': {e}. Resposta (limitada):\n'{response_text_debug[:500]}...'"); return err_payload
    except Exception as e: st.error(f"Erro (sent/temas) '{app_name}': {e}. Resposta (limitada):\n'{response_text_debug[:500]}...'"); return err_payload

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Extraindo features...")
def extract_feature_details_from_reviews(gemini_model_instance, app_name, review_texts_str):
    err_payload = {"elogiadas":[],"problematicas":[],"desejadas_ausentes":[{"funcionalidade":"Erro Análise","descricao_usuario":"Falha"}]}
    if not review_texts_str.strip(): return err_payload
    p = f"""Reviews de '{app_name}'. JSON: 'elogiadas','problematicas','desejadas_ausentes'. Cada: 'f','d'. Max 3-5/cat. REVIEWS: {str(review_texts_str)[:8000]} {COMMON_NAME_INSTRUCTION}"""
    try:
        r = gemini_model_instance.generate_content(p, generation_config=generation_config_json).text.strip()
        if not r: return err_payload
        return json.loads(r)
    except Exception as e: st.error(f"Erro (features) '{app_name}': {e}"); return err_payload

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Extraindo dor/encantamento...")
def extract_pain_delight_points_from_reviews(gemini_model_instance, app_name, review_texts_str):
    err_payload = {"pontos_dor":[{"tipo":"Erro","descricao":"Falha"}],"fatores_encantamento":[]}
    if not review_texts_str.strip(): return err_payload
    p = f"""Reviews de '{app_name}'. JSON: 'pontos_dor'(3-5 frustrações),'fatores_encantamento'(3-5 satisfações). Cada:'tipo','d','ex_review'. REVIEWS: {str(review_texts_str)[:8000]} {COMMON_NAME_INSTRUCTION}"""
    try:
        r = gemini_model_instance.generate_content(p, generation_config=generation_config_json).text.strip()
        if not r: return err_payload
        return json.loads(r)
    except Exception as e: st.error(f"Erro (dor/enc.) '{app_name}': {e}"); return err_payload

# ... (As funções de SÍNTESE e INSIGHTS de gráficos permanecem conceitualmente as mesmas, mas
# garantir que COMMON_NAME_INSTRUCTION esteja nos prompts e que não usem generation_config_json
# pois esperam Markdown, não JSON, como output direto.)

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Sintetizando GAPs de features...")
def synthesize_feature_gap_analysis(gemini_model_instance, all_apps_feature_details, my_app_name):
    # (Prompt como antes, com COMMON_NAME_INSTRUCTION)
    p = f"""Analista de produto. Features (incl. '{my_app_name}'): {json.dumps(all_apps_feature_details,ensure_ascii=False)}. GAPs/oportunidades (Markdown):
1. Forças '{my_app_name}'. 2. Fraquezas/GAPs '{my_app_name}'. 3. Oportunidades '{my_app_name}' (2-3). 4. Ameaças Concorrentes.{COMMON_NAME_INSTRUCTION}"""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (GAPs): {e}"); return f"Erro: {e}"


@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Sintetizando dor/encantamento...")
def synthesize_pain_delight_comparison(gemini_model_instance, all_apps_pain_delight_details, my_app_name):
    # (Prompt como antes, com COMMON_NAME_INSTRUCTION)
    p = f"""UX Expert. Dor/encantamento (incl. '{my_app_name}'): {json.dumps(all_apps_pain_delight_details,ensure_ascii=False)}. Análise comparativa (Markdown):
1. Dores Comuns. '{my_app_name}' sofre? 2. Dores Críticas '{my_app_name}'. 3. Encantamentos Diferenciais '{my_app_name}'. 4. Inspirações Concorrentes. 5. Recomendações '{my_app_name}'.{COMMON_NAME_INSTRUCTION}"""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (comp. dor/enc.): {e}"); return f"Erro: {e}"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sobre notas...")
def analyze_ratings_insights(gemini_model_instance, app_name, scores_distribution_str): # app_name aqui é o display_name completo
    p = f"App '{app_name}', dist. notas:\n{scores_distribution_str}\nParágrafo (máx 60 palavras) análise profissional. Use o nome '{app_name}' ao se referir ao app.{COMMON_NAME_INSTRUCTION}"
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (ratings) '{app_name}': {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sobre temas...")
def analyze_topics_insights(gemini_model_instance, app_name, topics_data_json_str): # app_name aqui é o display_name completo
    p = f"App '{app_name}', temas:\n{topics_data_json_str}\nParágrafo (máx 70 palavras) 1-2 temas +/- e implicações. Use o nome '{app_name}' ao se referir ao app.{COMMON_NAME_INSTRUCTION}"
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (topics) '{app_name}': {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sentimento comparativo...")
def analyze_comparative_sentiment_insights(gemini_model_instance, sentiment_comparison_str):
    p = f"Sentimento comparativo:\n{sentiment_comparison_str}\nParágrafo (máx 70 palavras) comparando perfis e implicações.{COMMON_NAME_INSTRUCTION}"
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (sent. comp.): {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando análise qualitativa estruturada...")
def generate_competitive_qualitative_analysis(gemini_model_instance, all_apps_sentiment_topic_analyses, my_app_name):
    # (Função modificada para usar generation_config_json para o output JSON)
    err_payload = {"analises_individuais":[{"app_name":my_app_name,"pontos_fortes":["Erro"],"pontos_fracos":[]}],"oportunidades_mercado":[],"ameacas_desafios_mercado":[],"tendencias_emergentes":[],"sugestoes_posicionamento_meu_app":[]}
    inp = "\n\n".join([f"App: {a['app_name']}\n- Sentimento: {json.dumps(a['sentiment_summary'])}\n- Temas: {json.dumps(a['top_topics'])}" for a in all_apps_sentiment_topic_analyses if "error" not in a.get("sentiment_summary",{})])
    if not inp: return err_payload
    p = f"""Analista de mercado. Dados de sentimento/temas (incl. '{my_app_name}'): {inp}
{COMMON_NAME_INSTRUCTION}
Retorne JSON com chaves: "analises_individuais":[{{"app_name","pontos_fortes":[],"pontos_fracos":[]}}],"oportunidades_mercado":[],"ameacas_desafios_mercado":[],"tendencias_emergentes":[],"sugestoes_posicionamento_meu_app":[] (para '{my_app_name}')."""
    try:
        r = gemini_model_instance.generate_content(p, generation_config=generation_config_json).text.strip()
        if not r: return err_payload
        return json.loads(r) # Não precisa mais de limpeza de ```json
    except Exception as e: st.error(f"Erro (análise qual.): {e}"); return err_payload

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando Matriz SWOT...")
def generate_swot_analysis(gemini_model_instance, my_app_name, my_app_strengths, my_app_weaknesses, market_opportunities, market_threats):
    # (Prompt como antes, com COMMON_NAME_INSTRUCTION)
    s,w,o,t = ["- "+"\n- ".join(l) if l else "Nenhuma informação específica identificada." for l in [my_app_strengths,my_app_weaknesses,market_opportunities,market_threats]]
    p = f"""Crie Matriz SWOT (Markdown) para '{my_app_name}'.
Forças:\n{s}\nFraquezas:\n{w}\nOportunidades:\n{o}\nAmeaças:\n{t}
Formato: ## Matriz SWOT para {my_app_name}\n### Forças\n- ...
{COMMON_NAME_INSTRUCTION}"""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (SWOT): {e}"); return "Erro SWOT."

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando perfil do público...")
def synthesize_audience_profile(gemini_model_instance, all_apps_collected_data_str, my_app_name_context):
    # (Prompt como antes, com COMMON_NAME_INSTRUCTION)
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

# --- Funções de Visualização (sem alterações de lógica interna, apenas garantindo que usam os nomes corretos) ---
# (Corpo completo das funções de plotagem como na versão anterior)
def plot_comparative_sentiment_chart(all_apps_sentiment_data): # (Corpo completo como na versão anterior)
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
    fig,ax=plt.subplots(figsize=(max(10,len(app_names)*1.5),6))
    df_filtered.plot(kind='bar',ax=ax,color=[colors[c] for c in df_filtered.columns])
    ax.set_title('Análise Comparativa de Sentimentos (%)',fontsize=15)
    ax.set_ylabel('Porcentagem de Reviews (%)');ax.set_xlabel('Aplicativos')
    ax.legend(title='Sentimento');ax.tick_params(axis='x',rotation=25,labelsize=10)
    plt.tight_layout();st.pyplot(fig)

def plot_topics_chart_for_app(app_analysis_data): # (Corpo completo como na versão anterior)
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
    fig,ax=plt.subplots(figsize=(8,max(4,len(df_topics)*0.5)))
    bar_colors_map={'positive':'#4CAF50','neutral':'#FFC107','negative':'#F44336'}
    df_plot=df_topics[['name','positive','neutral','negative']].set_index('name')
    df_plot.plot(kind='barh',stacked=True,color=[bar_colors_map['positive'],bar_colors_map['neutral'],bar_colors_map['negative']],ax=ax)
    ax.set_title(f'Principais Temas por Sentimento - {app_n}',fontsize=14)
    ax.set_xlabel('Nº de Menções');ax.set_ylabel('Tema')
    ax.tick_params(axis='x',colors='black');ax.tick_params(axis='y',colors='black',labelsize=9)
    ax.legend(['Positivo','Neutro','Negativo'],loc='lower right',frameon=False)
    plt.tight_layout();st.pyplot(fig)

def plot_ratings_distribution_chart(app_name,review_scores): # (Corpo completo como na versão anterior)
    if not review_scores: st.info(f"Nenhuma nota para {app_name}."); return "",None
    score_counts=Counter(review_scores);ratings_df=pd.DataFrame({'Estrelas':range(1,6)})
    counts_df=pd.DataFrame(list(score_counts.items()),columns=['Estrelas','Contagem']).sort_values(by='Estrelas')
    ratings_df=pd.merge(ratings_df,counts_df,on='Estrelas',how='left').fillna(0)
    ratings_df['Contagem']=ratings_df['Contagem'].astype(int)
    max_count_for_plot = max(1, ratings_df['Contagem'].max()) 

    fig,ax=plt.subplots(figsize=(7,5))
    bars=ax.bar(ratings_df['Estrelas'],ratings_df['Contagem'],color=['#F44336','#FF9800','#FFC107','#8BC34A','#4CAF50'])
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
    competitor_urls_st = [url for url in competitor_urls_st if url]
    analyze_button = st.button("🔍 Analisar Aplicativos", type="primary", use_container_width=True)

default_session_state = {'analysis_complete':False,'all_apps_processed_data':[],'feature_gap_report':"",'pain_delight_report':"",'overall_qualitative_data':{},'swot_report':"",'audience_profile_report':"",'my_app_name_for_synthesis':""}
for k,v in default_session_state.items():
    if k not in st.session_state: st.session_state[k]=v

if analyze_button:
    if not my_app_url_st: st.sidebar.warning("Insira a URL do seu app.")
    elif not model: st.error("Modelo Gemini não inicializado.")
    else:
        for k,v in default_session_state.items(): st.session_state[k]=v
        urls_to_process=[{"url":my_app_url_st,"role":"Meu App"}]
        urls_to_process.extend([{"url":url,"role":f"Concorrente {i+1}"} for i,url in enumerate(competitor_urls_st)])
        prog_bar=st.sidebar.progress(0); prog_text=st.sidebar.empty()
        total_steps=len(urls_to_process)*4+6 
        
        with st.spinner("Análise em progresso... pode levar minutos."):
            processed_data_list=[]
            step=0
            for i,app_info in enumerate(urls_to_process):
                app_id=get_app_id_from_url(app_info['url'])
                app_data_current={"display_name":f"{app_info['role']}:{app_id if app_id else 'Inv.'}","review_texts_str":"","review_scores":[],"sentiment_topic_analysis":{},"feature_details":{},"pain_delight_points":{}}
                if app_id:
                    step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) Coleta: {app_info['role']}")
                    texts,scores,official_name,short_name_id=fetch_play_store_data(app_id) # Modificado para pegar short_name_id
                    # Construção do nome de exibição: "Papel: Título Oficial (ID Curto se diferente do título)"
                    # Para concorrentes, priorizamos o nome curto extraído da URL, como solicitado.
                    if app_info['role'] == "Meu App":
                        d_name = f"{app_info['role']}: {official_name}"
                        if short_name_id and short_name_id.lower() not in official_name.lower():
                             d_name += f" ({short_name_id})"
                    else: # Concorrentes
                        # Se o título oficial for muito genérico ou igual ao do "Meu App", use o short_name_id
                        # Esta lógica pode precisar de mais refinamento dependendo dos casos reais.
                        # Por agora, vamos usar "Papel: Nome Curto (Título Oficial)" para dar mais destaque ao ID.
                        d_name = f"{app_info['role']}: {short_name_id}"
                        if official_name and official_name != app_id : # Não mostrar se for o próprio ID
                             d_name += f" ({official_name})"


                    app_data_current.update({"display_name":d_name,"review_scores":scores})
                    if app_info['role']=="Meu App":st.session_state.my_app_name_for_synthesis=d_name
                    
                    if texts:
                        r_str="\n".join(texts);app_data_current["review_texts_str"]=r_str
                        step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) Sent/Temas: {d_name[:30]}...")
                        app_data_current["sentiment_topic_analysis"]=analyze_single_app_reviews(model,r_str,d_name)
                        step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) Features: {d_name[:30]}...")
                        app_data_current["feature_details"]=extract_feature_details_from_reviews(model,d_name,r_str)
                        step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) Dor/Enc: {d_name[:30]}...")
                        app_data_current["pain_delight_points"]=extract_pain_delight_points_from_reviews(model,d_name,r_str)
                    else:app_data_current["sentiment_topic_analysis"]={"app_name":d_name,"sentiment_summary":{"no_reviews":100.0},"top_topics":[]}
                else:st.warning(f"URL inválida para {app_info['role']}. Pulando.")
                processed_data_list.append(app_data_current)
            st.session_state.all_apps_processed_data=processed_data_list

            if processed_data_list:
                my_app_n=st.session_state.my_app_name_for_synthesis
                if not my_app_n and processed_data_list:my_app_n=processed_data_list[0]["display_name"];st.session_state.my_app_name_for_synthesis=my_app_n
                
                step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) GAPs Features...")
                feat_l=[{"app_name":d["display_name"],**d["feature_details"]} for d in processed_data_list if d.get("feature_details")]
                if feat_l and my_app_n:st.session_state.feature_gap_report=synthesize_feature_gap_analysis(model,feat_l,my_app_n)

                step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) Comp. Dor/Enc...")
                pain_l=[{"app_name":d["display_name"],**d["pain_delight_points"]} for d in processed_data_list if d.get("pain_delight_points")]
                if pain_l and my_app_n:st.session_state.pain_delight_report=synthesize_pain_delight_comparison(model,pain_l,my_app_n)
                
                step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) Análise Qualitativa Geral...")
                sent_topic_l=[d["sentiment_topic_analysis"] for d in processed_data_list if d.get("sentiment_topic_analysis") and "no_reviews" not in d["sentiment_topic_analysis"].get("sentiment_summary",{})]
                if sent_topic_l and my_app_n:
                    st.session_state.overall_qualitative_data=generate_competitive_qualitative_analysis(model,sent_topic_l,my_app_n)
                    step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) SWOT {my_app_n[:25]}...")
                    q_data=st.session_state.overall_qualitative_data
                    my_app_q_an=next((item for item in q_data.get("analises_individuais",[]) if item.get("app_name")==my_app_n),None)
                    if my_app_q_an:st.session_state.swot_report=generate_swot_analysis(model,my_app_n,my_app_q_an.get("pontos_fortes",[]),my_app_q_an.get("pontos_fracos",[]),q_data.get("oportunidades_mercado",[]),q_data.get("ameacas_desafios_mercado",[]))
                
                step+=1;prog_bar.progress(min(1.0,step/total_steps));prog_text.info(f"({step}/{total_steps}) Perfil Público...")
                aud_prompt_data=[{"app_name":d["display_name"],"sentiment":d["sentiment_topic_analysis"].get("sentiment_summary"),"top_topics":d["sentiment_topic_analysis"].get("top_topics"),"requested_features":[f.get('funcionalidade') for f in d.get("feature_details",{}).get("desejadas_ausentes",[])],"pain_points":[p.get('descricao') for p in d.get("pain_delight_points",{}).get("pontos_dor",[])],"delight_factors":[dl.get('descricao') for dl in d.get("pain_delight_points",{}).get("fatores_encantamento",[])]} for d in processed_data_list]
                if aud_prompt_data and my_app_n:st.session_state.audience_profile_report=synthesize_audience_profile(model,json.dumps(aud_prompt_data,ensure_ascii=False,indent=2),my_app_n)
            
            st.session_state.analysis_complete=True
            prog_text.success("Análise Concluída!");prog_bar.progress(1.0)

if st.session_state.analysis_complete and st.session_state.all_apps_processed_data:
    st.header("🏁 Resultados da Análise Competitiva")
    tab_names = ["📊 Sentimento Geral","📱 Apps Individuais","🧩 GAPs Features","❤️ Dor vs. Enc.","♟️ SWOT","👤 Perfil Público","💡 Qualitativa Estratégica"]
    tabs=st.tabs(tab_names)
    my_app_n_disp = st.session_state.my_app_name_for_synthesis

    with tabs[0]: # Sentimento Geral
        st.subheader(tab_names[0])
        valid_sent=[d["sentiment_topic_analysis"] for d in st.session_state.all_apps_processed_data if d.get("sentiment_topic_analysis") and "no_reviews" not in d["sentiment_topic_analysis"].get("sentiment_summary",{}) and "error" not in d["sentiment_topic_analysis"].get("sentiment_summary",{})]
        if valid_sent:
            plot_comparative_sentiment_chart(valid_sent)
            sent_comp_str="\n".join([f"- App {s['app_name']}: Pos={s['sentiment_summary'].get('positive',0)}%,Neu={s['sentiment_summary'].get('neutral',0)}%,Neg={s['sentiment_summary'].get('negative',0)}%" for s in valid_sent])
            if sent_comp_str and model: st.markdown(f"**Análise IA:**\n{analyze_comparative_sentiment_insights(model,sent_comp_str)}")
        else: st.info("Dados insuficientes.")

    with tabs[1]: # Apps Individuais
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
    with tabs[4]:st.subheader(f"{tab_names[4]} ({my_app_n_disp if my_app_n_disp else 'Meu App'})");st.markdown(st.session_state.swot_report if st.session_state.swot_report else "Não gerado.") # Adicionado fallback para my_app_n_disp
    with tabs[5]:st.subheader(tab_names[5]);st.markdown(st.session_state.audience_profile_report if st.session_state.audience_profile_report else "Não gerado.")
    with tabs[6]:
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
