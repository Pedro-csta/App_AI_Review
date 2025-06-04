import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import google.generativeai as genai
from google_play_scraper import reviews, Sort, app as gp_app
from collections import Counter

# --- Configuração da Página Streamlit (Deve ser o primeiro comando Streamlit) ---
st.set_page_config(layout="wide", page_title="Análise de Concorrência de Apps")

# --- Configuração da API Key do Gemini e Constantes ---
gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
model = None

if not gemini_api_key:
    st.error("ERRO CRÍTICO: A chave API do Google Gemini (GOOGLE_API_KEY) não foi encontrada. "
             "Configure-a nos 'Secrets' do Streamlit Community Cloud.")
    st.stop()
else:
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Falha ao configurar o Gemini com a API Key. Erro: {e}")
        st.stop()

MAX_REVIEWS_TO_PROCESS = 500
APP_PLACEHOLDER_URL = "https://play.google.com/store/apps/details?id="

# --- Funções Auxiliares e de Extração de Dados ---
@st.cache_data(show_spinner=False)
def get_app_id_from_url(app_url):
    if not app_url or not app_url.startswith(APP_PLACEHOLDER_URL): return None
    try: return app_url.split('id=')[1].split('&')[0]
    except IndexError: return None

@st.cache_data(show_spinner="Buscando reviews e nome do app...")
def fetch_play_store_reviews_and_name(_app_id, lang='pt', country='br', count=MAX_REVIEWS_TO_PROCESS):
    if not _app_id: return [], [], _app_id
    app_name = _app_id
    try:
        app_details = gp_app(_app_id, lang=lang, country=country)
        app_name = app_details.get('title', _app_id)
    except Exception as e: st.warning(f"Não buscou detalhes do app {_app_id}. Usando ID. Erro: {e}")
    review_texts, review_scores = [], []
    try:
        app_reviews_data, _ = reviews(_app_id, lang=lang, country=country, sort=Sort.NEWEST, count=count, filter_score_with=None)
        for review in app_reviews_data:
            if review['content']: review_texts.append(review['content']); review_scores.append(review['score'])
        if not review_texts: st.info(f"Nenhum review para: {app_name} ({_app_id}).")
        return review_texts, review_scores, app_name
    except Exception as e: st.error(f"Erro ao buscar reviews para {app_name} ({_app_id}): {e}"); return [], [], app_name

# --- Funções de Análise com Gemini ---
@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Analisando sentimento e temas...")
def analyze_single_app_reviews(gemini_model_instance, reviews_text, app_name=""):
    if not reviews_text.strip(): return {"app_name": app_name, "sentiment_summary": {"no_reviews": 100.0}, "top_topics": []}
    num_reviews = len(reviews_text.split('\n')); ctx = f"do app '{app_name}'" if app_name else "de um app"
    p = f"""Analise os reviews {ctx}. {num_reviews} reviews. Formato JSON: {{ "app_name": "{app_name}", "sentiment_summary": {{...}}, "top_topics": [...] }}
sentiment_summary: % de Positivo, Neutro, Negativo, Sem Sentimento. Soma 100%.
top_topics: Lista de 5-7 temas (nome, e contagem de menções Positivas, Neutras, Negativas).
Ex: {{ "name": "Interface", "mentions": {{ "positive": 10, "negative": 2, "neutral": 3 }} }}
REVIEWS: {reviews_text}"""
    try:
        r = gemini_model_instance.generate_content(p).text.strip()
        if r.startswith("```json"): r = r[len("```json"):].strip()
        if r.endswith("```"): r = r[:-len("```")].strip()
        d = json.loads(r); s = d.get("sentiment_summary", {})
        if 'no_sentiment_detected' not in s: s['no_sentiment_detected'] = round(max(0,100.0 - sum(v for k,v in s.items() if k!='no_sentiment_detected')),2)
        ts = sum(s.values())
        if abs(ts - 100.0)>0.1 and ts!=0: s = {k:(v/ts*100 if ts>0 else 0) for k,v in s.items()} if ts>0 else {**s, 'no_sentiment_detected':100.0}
        d["sentiment_summary"] = {k:round(v,2) for k,v in s.items()}
        return d
    except Exception as e: st.error(f"Erro (sent/temas) '{app_name}': {e}"); return {"app_name":app_name, "sentiment_summary":{"error":100.0}, "top_topics":[]}

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Extraindo detalhes de funcionalidades...")
def extract_feature_details_from_reviews(gemini_model_instance, app_name, review_texts_str):
    if not review_texts_str.strip(): return {"elogiadas": [], "problematicas": [], "desejadas_ausentes": []}
    p = f"""Analise reviews de '{app_name}'. JSON: 'elogiadas', 'problematicas', 'desejadas_ausentes'. Cada: 'funcionalidade', 'descricao_usuario'. Max 3-5/cat. REVIEWS: {review_texts_str}"""
    try:
        r = gemini_model_instance.generate_content(p).text.strip()
        if r.startswith("```json"): r = r[len("```json"):].strip()
        if r.endswith("```"): r = r[:-len("```")].strip()
        return json.loads(r)
    except Exception as e: st.error(f"Erro (features) '{app_name}': {e}"); return {"elogiadas":[], "problematicas":[], "desejadas_ausentes":[{"funcionalidade":"Erro", "descricao_usuario":str(e)}]}

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Sintetizando GAPs de funcionalidades...")
def synthesize_feature_gap_analysis(gemini_model_instance, all_apps_feature_details, my_app_name):
    p = f"""Analista de produto. Detalhes de features (incluindo '{my_app_name}'): {json.dumps(all_apps_feature_details, ensure_ascii=False)}. Análise de GAPs/oportunidades (Markdown):
Use nomes completos dos apps (ex: 'Meu App: Nome XYZ').
1. Forças de '{my_app_name}'. 2. Fraquezas/GAPs de '{my_app_name}'. 3. Oportunidades para '{my_app_name}' (2-3). 4. Ameaças dos Concorrentes."""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (GAPs): {e}"); return f"Erro: {e}"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Extraindo dor/encantamento...")
def extract_pain_delight_points_from_reviews(gemini_model_instance, app_name, review_texts_str):
    if not review_texts_str.strip(): return {"pontos_dor": [], "fatores_encantamento": []}
    p = f"""Analise reviews de '{app_name}'. JSON: 'pontos_dor' (3-5 frustrações), 'fatores_encantamento' (3-5 satisfações). Cada: 'tipo', 'descricao', 'exemplo_review'. REVIEWS: {review_texts_str}"""
    try:
        r = gemini_model_instance.generate_content(p).text.strip()
        if r.startswith("```json"): r = r[len("```json"):].strip()
        if r.endswith("```"): r = r[:-len("```")].strip()
        return json.loads(r)
    except Exception as e: st.error(f"Erro (dor/enc.) '{app_name}': {e}"); return {"pontos_dor":[{"tipo":"Erro","descricao":str(e)}],"fatores_encantamento":[]}

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Sintetizando dor/encantamento...")
def synthesize_pain_delight_comparison(gemini_model_instance, all_apps_pain_delight_details, my_app_name):
    p = f"""UX Expert. Dor/encantamento (incluindo '{my_app_name}'): {json.dumps(all_apps_pain_delight_details, ensure_ascii=False)}. Análise comparativa (Markdown):
Use nomes completos dos apps.
1. Dores Comuns. '{my_app_name}' sofre? 2. Dores Críticas '{my_app_name}'. 3. Encantamentos Diferenciais '{my_app_name}'. 4. Inspirações dos Concorrentes. 5. Recomendações '{my_app_name}'."""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (comp. dor/enc.): {e}"); return f"Erro: {e}"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sobre notas...")
def analyze_ratings_insights(gemini_model_instance, app_name, scores_distribution_str):
    p = f"App '{app_name}', distribuição de notas:\n{scores_distribution_str}\nParágrafo (máx 60 palavras) de análise profissional. Use '{app_name}'."
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (ratings) '{app_name}': {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sobre temas...")
def analyze_topics_insights(gemini_model_instance, app_name, topics_data_json_str):
    p = f"App '{app_name}', temas:\n{topics_data_json_str}\nParágrafo (máx 70 palavras) 1-2 temas +/- e implicações. Use '{app_name}'."
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (topics) '{app_name}': {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando insights sentimento comparativo...")
def analyze_comparative_sentiment_insights(gemini_model_instance, sentiment_comparison_str):
    p = f"Sentimento comparativo:\n{sentiment_comparison_str}\nParágrafo (máx 70 palavras) comparando perfis e implicações. Use nomes completos dos apps."
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.warning(f"Erro (sent. comp.): {e}"); return "N/A"

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando análise qualitativa estruturada...")
def generate_competitive_qualitative_analysis(gemini_model_instance, all_apps_sentiment_topic_analyses, my_app_name):
    # Retorna JSON estruturado
    default_error_payload = {"analises_individuais": [{"app_name":my_app_name, "pontos_fortes":["Erro"], "pontos_fracos":[]}], "oportunidades_mercado": [], "ameacas_desafios_mercado": [], "tendencias_emergentes": [], "sugestoes_posicionamento_meu_app": []}
    inp = "\n\n".join([f"App: {a['app_name']}\n- Sentimento: {json.dumps(a['sentiment_summary'])}\n- Temas: {json.dumps(a['top_topics'])}" for a in all_apps_sentiment_topic_analyses if "error" not in a.get("sentiment_summary",{})])
    if not inp: return default_error_payload
    p = f"""Analista de mercado. Dados de sentimento/temas (incluindo '{my_app_name}'): {inp}
Use nomes completos dos apps. Retorne JSON com chaves:
"analises_individuais": [{{ "app_name", "pontos_fortes": [], "pontos_fracos": [] }}],
"oportunidades_mercado": [], "ameacas_desafios_mercado": [],
"tendencias_emergentes": [] (2-3), "sugestoes_posicionamento_meu_app": [] (1-2 para '{my_app_name}')."""
    try:
        r = gemini_model_instance.generate_content(p).text.strip()
        if r.startswith("```json"): r = r[len("```json"):].strip()
        if r.endswith("```"): r = r[:-len("```")].strip()
        return json.loads(r)
    except Exception as e: st.error(f"Erro (análise qual.): {e}"); return default_error_payload

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando Matriz SWOT...")
def generate_swot_analysis(gemini_model_instance, my_app_name, my_app_strengths, my_app_weaknesses, market_opportunities, market_threats):
    s = "- " + "\n- ".join(my_app_strengths) if my_app_strengths else "N/A."
    w = "- " + "\n- ".join(my_app_weaknesses) if my_app_weaknesses else "N/A."
    o = "- " + "\n- ".join(market_opportunities) if market_opportunities else "N/A."
    t = "- " + "\n- ".join(market_threats) if market_threats else "N/A."
    p = f"""Crie Matriz SWOT (Markdown) para '{my_app_name}'.
Forças:\n{s}\nFraquezas:\n{w}\nOportunidades:\n{o}\nAmeaças:\n{t}
Formato: ## Matriz SWOT para {my_app_name}\n### Forças\n- ..."""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (SWOT): {e}"); return "Erro SWOT."

@st.cache_data(hash_funcs={genai.GenerativeModel: lambda _: None}, show_spinner="Gerando perfil do público...")
def synthesize_audience_profile(gemini_model_instance, all_apps_collected_data_str, my_app_name_to_mention):
    p = f"""Baseado nos reviews de '{my_app_name_to_mention}' e concorrentes, crie perfil do público (Markdown). Use nomes completos dos apps.
* Visão Geral: Quem são? Objetivos?
* Dores Comuns?
* O que Valorizam?
* Objeções/Preocupações?
* Perfil de Uso?
Dados: {all_apps_collected_data_str}"""
    try: return gemini_model_instance.generate_content(p).text.strip()
    except Exception as e: st.error(f"Erro (perfil público): {e}"); return "Erro perfil público."

# --- Funções de Visualização (sem alterações de lógica interna) ---
def plot_comparative_sentiment_chart(all_apps_sentiment_data):
    # (Corpo completo como na versão anterior)
    if not all_apps_sentiment_data: st.info("Nenhum dado de sentimento para plotar."); return
    plot_data = {} # Recriando para brevidade
    categories = ['positive', 'neutral', 'negative', 'no_sentiment_detected']
    display_labels = {'positive': 'Positivo', 'neutral': 'Neutro', 'negative': 'Negativo', 'no_sentiment_detected': 'Não Detectado'}
    app_names = []
    valid_data_found = False
    for data_item in all_apps_sentiment_data:
        if data_item and "app_name" in data_item and "sentiment_summary" in data_item and "error" not in data_item["sentiment_summary"]:
            valid_data_found = True; app_name = data_item['app_name']; app_names.append(app_name)
            for cat in categories:
                if cat not in plot_data: plot_data[cat] = []
                plot_data[cat].append(data_item['sentiment_summary'].get(cat, 0))
    if not valid_data_found: st.info("Nenhum app com dados de sentimento válidos."); return
    df = pd.DataFrame(plot_data, index=app_names).rename(columns=display_labels)
    colors = {'Positivo': '#4CAF50', 'Neutro': '#FFC107', 'Negativo': '#F44336', 'Não Detectado': '#9E9E9E'}
    df_filtered = df[[col for col in df.columns if df[col].sum() > 0 and col in colors]]
    if df_filtered.empty: st.info("Dados de sentimento zerados."); return
    fig, ax = plt.subplots(figsize=(max(10, len(app_names) * 1.5), 6))
    df_filtered.plot(kind='bar', ax=ax, color=[colors[col] for col in df_filtered.columns])
    ax.set_title('Análise Comparativa de Sentimentos (%)', fontsize=15)
    ax.set_ylabel('Porcentagem de Reviews (%)'); ax.set_xlabel('Aplicativos')
    ax.legend(title='Sentimento'); ax.tick_params(axis='x', rotation=25, labelsize=10)
    plt.tight_layout(); st.pyplot(fig)

def plot_topics_chart_for_app(app_analysis_data):
    # (Corpo completo como na versão anterior)
    app_name = app_analysis_data.get('app_name', 'App Desconhecido')
    topics_data = app_analysis_data.get('top_topics', [])
    if not topics_data: st.info(f"Nenhum tema para {app_name}."); return
    parsed_topics = []
    for topic in topics_data:
        if "name" in topic and "mentions" in topic:
            parsed_topics.append({'name': topic['name'], 'positive': topic['mentions'].get('positive', 0),
                                  'neutral': topic['mentions'].get('neutral', 0), 'negative': topic['mentions'].get('negative', 0)})
    if not parsed_topics: st.info(f"Temas malformatados para {app_name}."); return
    df_topics = pd.DataFrame(parsed_topics); df_topics['Total'] = df_topics['positive'] + df_topics['neutral'] + df_topics['negative']
    df_topics = df_topics[df_topics['Total'] > 0]
    if df_topics.empty: st.info(f"Nenhum tema com menções para {app_name}."); return
    df_topics = df_topics.sort_values('Total', ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_topics) * 0.5)))
    bar_colors_map = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    df_to_plot = df_topics[['name', 'positive', 'neutral', 'negative']].set_index('name')
    df_to_plot.plot(kind='barh', stacked=True, color=[bar_colors_map['positive'], bar_colors_map['neutral'], bar_colors_map['negative']], ax=ax)
    ax.set_title(f'Principais Temas por Sentimento - {app_name}', fontsize=14)
    ax.set_xlabel('Nº de Menções'); ax.set_ylabel('Tema')
    ax.tick_params(axis='x', colors='black'); ax.tick_params(axis='y', colors='black', labelsize=9)
    ax.legend(['Positivo', 'Neutro', 'Negativo'], loc='lower right', frameon=False)
    plt.tight_layout(); st.pyplot(fig)

def plot_ratings_distribution_chart(app_name, review_scores):
    # (Corpo completo como na versão anterior)
    if not review_scores: st.info(f"Nenhuma nota para {app_name}."); return "", None
    score_counts = Counter(review_scores)
    ratings_df = pd.DataFrame({'Estrelas': range(1, 6)})
    counts_df = pd.DataFrame(list(score_counts.items()), columns=['Estrelas', 'Contagem']).sort_values(by='Estrelas')
    ratings_df = pd.merge(ratings_df, counts_df, on='Estrelas', how='left').fillna(0)
    ratings_df['Contagem'] = ratings_df['Contagem'].astype(int)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(ratings_df['Estrelas'], ratings_df['Contagem'], color=['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50'])
    ax.set_title(f'Distribuição de Notas - {app_name}', fontsize=14)
    ax.set_xlabel('Nota (Estrelas)'); ax.set_ylabel('Nº de Reviews')
    ax.set_xticks(range(1, 6))
    for bar in bars:
        yval = bar.get_height()
        if yval > 0: plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(1,max(ratings_df['Contagem'])), int(yval), ha='center', va='bottom')
    plt.tight_layout(); st.pyplot(fig)
    total_reviews_with_score = sum(ratings_df['Contagem'])
    if total_reviews_with_score == 0: return "Nenhuma nota válida.", None
    distribution_str = f"Total de reviews com notas: {total_reviews_with_score}.\n"
    distribution_str += "\n".join([f"- {row['Estrelas']} estrela(s): {row['Contagem']} reviews ({row['Contagem']*100/total_reviews_with_score:.1f}%)" for index, row in ratings_df.iterrows()])
    return distribution_str, ratings_df

# --- Interface Streamlit e Lógica Principal ---
st.title("🤖 Ferramenta Avançada de Análise de Concorrência de Apps")
st.markdown(f"Insira as URLs da Play Store para seu app e até 3 concorrentes. Serão analisados até **{MAX_REVIEWS_TO_PROCESS} reviews** por app.")

with st.sidebar:
    st.header("🔗 URLs dos Aplicativos")
    my_app_url_st = st.text_input("URL do Seu App (Foco Central):", placeholder=APP_PLACEHOLDER_URL + "com.example.myapp", key="my_app_url_input_st")
    competitor_urls_st = []
    for i in range(3):
        url = st.text_input(f"URL Concorrente {i+1}:", placeholder=APP_PLACEHOLDER_URL + f"com.example.competitor{i+1}", key=f"comp{i}_url_st")
        if url: competitor_urls_st.append(url)
    analyze_button = st.button("🔍 Analisar Aplicativos", type="primary", use_container_width=True)

# Inicialização do Estado da Sessão
if 'analysis_complete' not in st.session_state: st.session_state.analysis_complete = False
if 'all_apps_processed_data' not in st.session_state: st.session_state.all_apps_processed_data = []
if 'feature_gap_report' not in st.session_state: st.session_state.feature_gap_report = ""
if 'pain_delight_report' not in st.session_state: st.session_state.pain_delight_report = ""
if 'overall_qualitative_data' not in st.session_state: st.session_state.overall_qualitative_data = {} # Para o JSON
if 'swot_report' not in st.session_state: st.session_state.swot_report = ""
if 'audience_profile_report' not in st.session_state: st.session_state.audience_profile_report = ""
if 'my_app_name_for_synthesis_st' not in st.session_state: st.session_state.my_app_name_for_synthesis_st = ""

if analyze_button:
    if not my_app_url_st: st.sidebar.warning("Por favor, insira a URL do seu app.")
    elif not model: st.error("Modelo Gemini não inicializado. Verifique a API Key.")
    else:
        # Resetar estado para nova análise
        for key in ['analysis_complete', 'all_apps_processed_data', 'feature_gap_report', 
                    'pain_delight_report', 'overall_qualitative_data', 'swot_report', 
                    'audience_profile_report', 'my_app_name_for_synthesis_st']:
            if key.endswith('_report') or key.endswith('_st'): st.session_state[key] = ""
            elif key.endswith('_data'): st.session_state[key] = [] if key == 'all_apps_processed_data' else {}
            else: st.session_state[key] = False
        
        urls_to_process_st = [{"url": my_app_url_st, "role": "Meu App"}]
        for i, comp_url in enumerate(competitor_urls_st): urls_to_process_st.append({"url": comp_url, "role": f"Concorrente {i+1}"})

        progress_bar = st.sidebar.progress(0)
        progress_status_text = st.sidebar.empty()
        # fetch + 3 análises IA por app + 5 sínteses globais
        total_steps = len(urls_to_process_st) * (1 + 3) + 5 
        
        with st.spinner("Iniciando análise completa... Por favor, aguarde."):
            processed_data_list = []
            step_counter = 0

            for i, app_info in enumerate(urls_to_process_st):
                display_role_url = f"{app_info['role']} ({app_info['url'][:50]}...)"
                app_id_st = get_app_id_from_url(app_info['url'])
                current_app_proc_data = {"display_name": f"{app_info['role']}: {app_id_st if app_id_st else 'ID Inválido'}", "review_texts_str": "", "review_scores": [], "sentiment_topic_analysis": {}, "feature_details": {}, "pain_delight_points": {}}
                
                if app_id_st:
                    step_counter += 1; progress_bar.progress(min(1.0, step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Coletando: {app_info['role']}")
                    texts, scores, name = fetch_play_store_reviews_and_name(app_id_st) # count usa MAX_REVIEWS_TO_PROCESS
                    d_name = f"{app_info['role']}: {name}" if app_info['role']!=name else name
                    current_app_proc_data.update({"display_name": d_name, "review_scores": scores})
                    if app_info['role'] == "Meu App": st.session_state.my_app_name_for_synthesis_st = d_name
                    
                    if texts:
                        r_str = "\n".join(texts); current_app_proc_data["review_texts_str"] = r_str
                        step_counter+=1; progress_bar.progress(min(1.0,step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Sent/Temas: {d_name}")
                        current_app_proc_data["sentiment_topic_analysis"] = analyze_single_app_reviews(model, r_str, d_name)
                        step_counter+=1; progress_bar.progress(min(1.0,step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Features: {d_name}")
                        current_app_proc_data["feature_details"] = extract_feature_details_from_reviews(model, d_name, r_str)
                        step_counter+=1; progress_bar.progress(min(1.0,step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Dor/Enc: {d_name}")
                        current_app_proc_data["pain_delight_points"] = extract_pain_delight_points_from_reviews(model, d_name, r_str)
                    else: # Sem reviews
                        current_app_proc_data["sentiment_topic_analysis"] = {"app_name": d_name, "sentiment_summary": {"no_reviews": 100.0}, "top_topics": []}
                        # Outras análises também ficam vazias
                else: st.warning(f"URL inválida para {app_info['role']}. Pulando.")
                processed_data_list.append(current_app_proc_data)
            st.session_state.all_apps_processed_data = processed_data_list

            if processed_data_list:
                my_app_name = st.session_state.my_app_name_for_synthesis_st
                if not my_app_name and processed_data_list: my_app_name = processed_data_list[0]["display_name"] # Fallback
                
                step_counter+=1; progress_bar.progress(min(1.0,step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Gerando GAPs de Features...")
                feat_list = [{"app_name":d["display_name"], **d["feature_details"]} for d in processed_data_list if d.get("feature_details")]
                if feat_list and my_app_name: st.session_state.feature_gap_report = synthesize_feature_gap_analysis(model, feat_list, my_app_name)

                step_counter+=1; progress_bar.progress(min(1.0,step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Comparando Dor/Encantamento...")
                pain_list = [{"app_name":d["display_name"], **d["pain_delight_points"]} for d in processed_data_list if d.get("pain_delight_points")]
                if pain_list and my_app_name: st.session_state.pain_delight_report = synthesize_pain_delight_comparison(model, pain_list, my_app_name)
                
                step_counter+=1; progress_bar.progress(min(1.0,step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Análise Qualitativa Geral...")
                sent_topic_list = [d["sentiment_topic_analysis"] for d in processed_data_list if d.get("sentiment_topic_analysis") and "no_reviews" not in d["sentiment_topic_analysis"].get("sentiment_summary", {})]
                if sent_topic_list and my_app_name:
                    st.session_state.overall_qualitative_data = generate_competitive_qualitative_analysis(model, sent_topic_list, my_app_name)
                    
                    # Gerar SWOT com base na análise qualitativa
                    step_counter+=1; progress_bar.progress(min(1.0,step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Gerando SWOT para {my_app_name}...")
                    qual_data = st.session_state.overall_qualitative_data
                    my_app_qual_an = next((item for item in qual_data.get("analises_individuais", []) if item.get("app_name") == my_app_name), None)
                    if my_app_qual_an:
                        st.session_state.swot_report = generate_swot_analysis(model, my_app_name, my_app_qual_an.get("pontos_fortes",[]), my_app_qual_an.get("pontos_fracos",[]), qual_data.get("oportunidades_mercado",[]), qual_data.get("ameacas_desafios_mercado",[]))
                
                step_counter+=1; progress_bar.progress(min(1.0,step_counter/total_steps)); progress_status_text.info(f"({step_counter}/{total_steps}) Gerando Perfil do Público...")
                audience_prompt_data = []
                for d in processed_data_list:
                    audience_prompt_data.append({"app_name": d["display_name"], "sentiment": d["sentiment_topic_analysis"].get("sentiment_summary"), "top_topics": d["sentiment_topic_analysis"].get("top_topics"), "requested_features": [f.get('funcionalidade') for f in d["feature_details"].get("desejadas_ausentes", [])], "pain_points": [p.get('descricao') for p in d["pain_delight_points"].get("pontos_dor", [])], "delight_factors": [dl.get('descricao') for dl in d["pain_delight_points"].get("fatores_encantamento", [])]})
                if audience_prompt_data and my_app_name: st.session_state.audience_profile_report = synthesize_audience_profile(model, json.dumps(audience_prompt_data, ensure_ascii=False, indent=2), my_app_name)

            st.session_state.analysis_complete = True
            progress_status_text.success("Análise Concluída!")
            progress_bar.progress(1.0)

if st.session_state.analysis_complete and st.session_state.all_apps_processed_data:
    st.header("🏁 Resultados da Análise Competitiva")
    tab_titles = ["📊 Sentimento Geral", "📱 Apps Individuais", "🧩 GAPs de Features", "❤️ Dor vs. Encantamento", "♟️ Matriz SWOT", "👤 Perfil do Público", "💡 Qualitativa Estratégica"]
    tabs = st.tabs(tab_titles)

    with tabs[0]: # Sentimento Geral
        st.subheader(tab_titles[0])
        # ... (código de display como antes)
        valid_sent_analyses = [d["sentiment_topic_analysis"] for d in st.session_state.all_apps_processed_data if d.get("sentiment_topic_analysis") and "no_reviews" not in d["sentiment_topic_analysis"].get("sentiment_summary",{}) and "error" not in d["sentiment_topic_analysis"].get("sentiment_summary",{})]
        if valid_sent_analyses:
            plot_comparative_sentiment_chart(valid_sent_analyses)
            sent_comp_str = "\n".join([f"- App: {s['app_name']}, Sentimento: Positivo={s['sentiment_summary'].get('positive',0)}%, Neutro={s['sentiment_summary'].get('neutral',0)}%, Negativo={s['sentiment_summary'].get('negative',0)}%" for s in valid_sent_analyses])
            if sent_comp_str and model: st.markdown(f"**Análise Profissional (Sentimento Comparativo):**\n{analyze_comparative_sentiment_insights(model, sent_comp_str)}")
        else: st.info("Não há dados de sentimento suficientes.")


    with tabs[1]: # Apps Individuais
        st.subheader(tab_titles[1])
        # ... (código de display como antes, com expanders)
        for app_data_st in st.session_state.all_apps_processed_data:
            display_name_st_tab = app_data_st["display_name"]
            is_my_app = st.session_state.my_app_name_for_synthesis_st == display_name_st_tab
            with st.expander(f"Ver detalhes para: {display_name_st_tab}", expanded=is_my_app):
                # Notas
                st.markdown(f"**Distribuição de Notas**")
                if app_data_st["review_scores"]:
                    ratings_dist_str_st, _ = plot_ratings_distribution_chart(display_name_st_tab, app_data_st["review_scores"])
                    if ratings_dist_str_st and "Nenhuma nota válida" not in ratings_dist_str_st and model: st.markdown(f"**Análise (Notas):**\n{analyze_ratings_insights(model, display_name_st_tab, ratings_dist_str_st)}")
                else: st.caption(f"Sem dados de notas.")
                st.markdown("---")
                # Temas
                st.markdown(f"**Principais Temas**")
                s_t_analysis_st = app_data_st["sentiment_topic_analysis"]
                if s_t_analysis_st and "no_reviews" not in s_t_analysis_st.get("sentiment_summary",{}) and "error" not in s_t_analysis_st.get("sentiment_summary",{}) and s_t_analysis_st.get("top_topics"):
                    plot_topics_chart_for_app(s_t_analysis_st)
                    topics_json_str_st = json.dumps(s_t_analysis_st.get("top_topics", []), ensure_ascii=False, indent=2)
                    if model: st.markdown(f"**Análise (Temas):**\n{analyze_topics_insights(model, display_name_st_tab, topics_json_str_st)}")
                    with st.popover("Dados brutos dos temas"): st.json(s_t_analysis_st.get("top_topics", []))
                else: st.caption(f"Sem dados de temas ou reviews.")
                st.markdown("---")
                # Features
                st.markdown(f"**Detalhes de Funcionalidades (Extraído)**")
                if app_data_st.get("feature_details") and (app_data_st["feature_details"].get("elogiadas") or app_data_st["feature_details"].get("problematicas") or app_data_st["feature_details"].get("desejadas_ausentes")):
                    st.json(app_data_st["feature_details"])
                else: st.caption(f"Não foram extraídos detalhes de funcionalidades.")
                st.markdown("---")
                # Dor/Encantamento
                st.markdown(f"**Pontos de Dor e Encantamento (Extraído)**")
                if app_data_st.get("pain_delight_points") and (app_data_st["pain_delight_points"].get("pontos_dor") or app_data_st["pain_delight_points"].get("fatores_encantamento")):
                    st.json(app_data_st["pain_delight_points"])
                else: st.caption(f"Não foram extraídos pontos de dor/encantamento.")


    with tabs[2]: # GAPs de Features
        st.subheader(tab_titles[2])
        if st.session_state.feature_gap_report: st.markdown(st.session_state.feature_gap_report)
        else: st.info("Relatório de GAPs não gerado.")

    with tabs[3]: # Dor vs. Encantamento
        st.subheader(tab_titles[3])
        if st.session_state.pain_delight_report: st.markdown(st.session_state.pain_delight_report)
        else: st.info("Comparativo de Dor/Encantamento não gerado.")
            
    with tabs[4]: # Matriz SWOT
        st.subheader(f"{tab_titles[4]} para {st.session_state.my_app_name_for_synthesis_st}")
        if st.session_state.swot_report: st.markdown(st.session_state.swot_report)
        else: st.info("Matriz SWOT não gerada.")

    with tabs[5]: # Perfil do Público
        st.subheader(tab_titles[5])
        if st.session_state.audience_profile_report: st.markdown(st.session_state.audience_profile_report)
        else: st.info("Perfil do Público não gerado.")
            
    with tabs[6]: # Qualitativa Estratégica
        st.subheader(tab_titles[6])
        qual_data = st.session_state.get("overall_qualitative_data", {})
        if qual_data and qual_data.get("analises_individuais"): # Verifica se há dados
            st.markdown("#### Análises Individuais (Resumo Estratégico)")
            for app_an in qual_data.get("analises_individuais", []):
                st.markdown(f"**App: {app_an.get('app_name')}**")
                st.markdown(f"* **Pontos Fortes Clave:** {'; '.join(app_an.get('pontos_fortes', ['N/A']))}")
                st.markdown(f"* **Pontos Fracos Clave:** {'; '.join(app_an.get('pontos_fracos', ['N/A']))}")
                st.markdown("---")
            
            st.markdown("#### Oportunidades de Mercado Identificadas")
            st.markdown("\n".join([f"- {op}" for op in qual_data.get("oportunidades_mercado", ["N/A"])]))
            
            st.markdown("#### Ameaças e Desafios de Mercado Comuns")
            st.markdown("\n".join([f"- {am}" for am in qual_data.get("ameacas_desafios_mercado", ["N/A"])]))

            st.markdown("#### Tendências Emergentes Observadas")
            st.markdown("\n".join([f"- {te}" for te in qual_data.get("tendencias_emergentes", ["N/A"])]))

            my_app_name_to_show = st.session_state.my_app_name_for_synthesis_st if st.session_state.my_app_name_for_synthesis_st else "Seu App"
            st.markdown(f"#### Sugestões de Posicionamento Estratégico para {my_app_name_to_show}")
            st.markdown("\n".join([f"- {sp}" for sp in qual_data.get("sugestoes_posicionamento_meu_app", ["N/A"])]))
        else:
            st.info("Análise qualitativa geral não gerada ou dados insuficientes.")

elif not analyze_button: # Estado inicial antes do primeiro clique
    st.info("⬅️ Insira as URLs dos apps na barra lateral e clique em 'Analisar Aplicativos' para começar.")

st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido com IA Avançada ✨")
