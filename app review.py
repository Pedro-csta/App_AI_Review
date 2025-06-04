
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
             "Configure-a nos 'Secrets' do Streamlit Community Cloud (recomendado para deploy).")
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
    if not app_url or not app_url.startswith(APP_PLACEHOLDER_URL):
        return None
    try:
        return app_url.split('id=')[1].split('&')[0]
    except IndexError:
        return None

@st.cache_data(show_spinner="Buscando reviews e nome do app...")
def fetch_play_store_reviews_and_name(_app_id, lang='pt', country='br', count=MAX_REVIEWS_TO_PROCESS):
    if not _app_id: return [], [], _app_id
    app_name = _app_id
    try:
        app_details = gp_app(_app_id, lang=lang, country=country)
        app_name = app_details.get('title', _app_id)
    except Exception as e:
        st.warning(f"Não foi possível buscar detalhes do app {_app_id}. Usando ID. Erro: {e}")
    review_texts, review_scores = [], []
    try:
        app_reviews_data, _ = reviews(_app_id, lang=lang, country=country, sort=Sort.NEWEST, count=count, filter_score_with=None)
        for review in app_reviews_data:
            if review['content']: review_texts.append(review['content']); review_scores.append(review['score'])
        if not review_texts: st.info(f"Nenhum review encontrado para: {app_name} ({_app_id}).")
        return review_texts, review_scores, app_name
    except Exception as e:
        st.error(f"Erro ao buscar reviews para {app_name} ({_app_id}): {e}")
        return [], [], app_name

# --- Funções de Análise com Gemini ---
@st.cache_data(show_spinner="Analisando sentimento e temas...")
def analyze_single_app_reviews(gemini_model, reviews_text, app_name=""):
    if not reviews_text.strip():
        return {"app_name": app_name, "sentiment_summary": {"no_reviews": 100.0}, "top_topics": []}
    num_reviews_in_prompt = len(reviews_text.split('\n'))
    app_context = f"do app '{app_name}'" if app_name else "de um app"
    prompt = f"""Analise os reviews {app_context} da Play Store. {num_reviews_in_prompt} reviews de amostra.
Formato JSON: {{ "app_name": "{app_name}", "sentiment_summary": {{...}}, "top_topics": [...] }}
sentiment_summary: % de Positivo, Neutro, Negativo, Sem Sentimento. Soma 100%.
top_topics: Lista de 5-7 temas (nome, e contagem de menções Positivas, Neutras, Negativas).
Exemplo de tema: {{ "name": "Interface", "mentions": {{ "positive": 10, "negative": 2, "neutral": 3 }} }}
REVIEWS: ---BEGIN REVIEWS--- {reviews_text} ---END REVIEWS---"""
    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        if response_text.startswith("```json"): response_text = response_text[len("```json"):].strip()
        if response_text.endswith("```"): response_text = response_text[:-len("```")].strip()
        data = json.loads(response_text)
        sent_data = data.get("sentiment_summary", {})
        if 'no_sentiment_detected' not in sent_data:
            current_total = sum(v for k,v in sent_data.items() if k != 'no_sentiment_detected')
            sent_data['no_sentiment_detected'] = round(max(0,100.0 - current_total),2)
        total_sum = sum(sent_data.values())
        if abs(total_sum - 100.0) > 0.1 and total_sum != 0:
            if total_sum > 0:
                for key_s in sent_data: sent_data[key_s] = round(sent_data[key_s]/total_sum*100,2)
            else: sent_data = {k:0.0 for k in sent_data}; sent_data['no_sentiment_detected'] = 100.0
        data["sentiment_summary"] = sent_data
        return data
    except Exception as e:
        st.error(f"Erro (analyze_single_app_reviews) para '{app_name}': {e}")
        return {"app_name": app_name, "sentiment_summary": {"error": 100.0}, "top_topics": []}

@st.cache_data(show_spinner="Extraindo detalhes de funcionalidades...")
def extract_feature_details_from_reviews(gemini_model, app_name, review_texts_str):
    if not review_texts_str.strip(): return {"elogiadas": [], "problematicas": [], "desejadas_ausentes": []}
    prompt = f"""Analise os reviews para '{app_name}'. Identifique em JSON:
1.  'elogiadas': Recursos positivos.
2.  'problematicas': Recursos com problemas/bugs.
3.  'desejadas_ausentes': Recursos solicitados/ausentes.
Para cada: 'funcionalidade' (nome) e 'descricao_usuario' (resumo). Max 3-5 itens/categoria.
Exemplo: {{ "funcionalidade": "Backup Nuvem", "descricao_usuario": "Backup funciona sem falhas." }}
Reviews: ---BEGIN REVIEWS--- {review_texts_str} ---END REVIEWS---"""
    try:
        response = gemini_model.generate_content(prompt); response_text = response.text.strip()
        if response_text.startswith("```json"): response_text = response_text[len("```json"):].strip()
        if response_text.endswith("```"): response_text = response_text[:-len("```")].strip()
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Erro (extract_feature_details) para '{app_name}': {e}")
        return {"elogiadas": [], "problematicas": [], "desejadas_ausentes": [{"funcionalidade":"Erro na Análise", "descricao_usuario":str(e)}]}

@st.cache_data(show_spinner="Sintetizando análise de GAPs...")
def synthesize_feature_gap_analysis(gemini_model, all_apps_feature_details, my_app_name):
    all_apps_feature_details_json_str = json.dumps(all_apps_feature_details, ensure_ascii=False, indent=2)
    prompt = f"""Analista de produto. Baseado nos detalhes de funcionalidades (incluindo '{my_app_name}'), forneça análise de GAPs/oportunidades em Markdown.
Dados: {all_apps_feature_details_json_str}
Sua análise deve cobrir:
1.  **Forças de '{my_app_name}'**: Funcionalidades elogiadas em '{my_app_name}', especialmente se problemáticas nos concorrentes.
2.  **Fraquezas/GAPs de '{my_app_name}'**: Funcionalidades problemáticas em '{my_app_name}' ou desejadas/ausentes que são elogiadas nos concorrentes.
3.  **Oportunidades Chave para '{my_app_name}'** (2-3 prioritárias).
4.  **Ameaças dos Concorrentes**: Funcionalidades onde concorrentes são mais fortes."""
    try: return gemini_model.generate_content(prompt).text.strip()
    except Exception as e: st.error(f"Erro (synthesize_feature_gap): {e}"); return f"Erro: {e}"

@st.cache_data(show_spinner="Extraindo pontos de dor/encantamento...")
def extract_pain_delight_points_from_reviews(gemini_model, app_name, review_texts_str):
    if not review_texts_str.strip(): return {"pontos_dor": [], "fatores_encantamento": []}
    prompt = f"""Analise os reviews para '{app_name}'. Identifique em JSON:
1.  'pontos_dor': 3-5 experiências de frustração.
2.  'fatores_encantamento': 3-5 experiências de satisfação/lealdade.
Para cada: 'tipo', 'descricao' e 'exemplo_review' (anônimo).
Exemplo: {{ "tipo": "Ponto de Dor", "descricao": "Cadastro longo.", "exemplo_review": "Cadastro pede muita info." }}
Reviews: ---BEGIN REVIEWS--- {review_texts_str} ---END REVIEWS---"""
    try:
        response = gemini_model.generate_content(prompt); response_text = response.text.strip()
        if response_text.startswith("```json"): response_text = response_text[len("```json"):].strip()
        if response_text.endswith("```"): response_text = response_text[:-len("```")].strip()
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Erro (extract_pain_delight) para '{app_name}': {e}")
        return {"pontos_dor": [{"tipo":"Erro", "descricao":str(e)}], "fatores_encantamento": []}

@st.cache_data(show_spinner="Sintetizando comparação de dor/encantamento...")
def synthesize_pain_delight_comparison(gemini_model, all_apps_pain_delight_details, my_app_name):
    all_apps_pain_delight_details_json_str = json.dumps(all_apps_pain_delight_details, ensure_ascii=False, indent=2)
    prompt = f"""Especialista em UX. Baseado nos pontos de dor/encantamento (incluindo '{my_app_name}'), forneça análise comparativa em Markdown.
Dados: {all_apps_pain_delight_details_json_str}
Análise:
1.  **Pontos de Dor Comuns no Mercado**. '{my_app_name}' sofre?
2.  **Pontos de Dor Críticos para '{my_app_name}'**.
3.  **Fatores de Encantamento Diferenciais de '{my_app_name}'**.
4.  **Inspirações dos Concorrentes**.
5.  **Recomendações para '{my_app_name}'** (1-2 sugestões)."""
    try: return gemini_model.generate_content(prompt).text.strip()
    except Exception as e: st.error(f"Erro (synthesize_pain_delight): {e}"); return f"Erro: {e}"

@st.cache_data(show_spinner="Gerando insights sobre notas...")
def analyze_ratings_insights(gemini_model, app_name, scores_distribution_str):
    prompt = f"App '{app_name}', distribuição de notas:\n{scores_distribution_str}\nForneça um parágrafo (máx 60 palavras) de análise profissional sobre o que isso sugere."
    try: return gemini_model.generate_content(prompt).text.strip()
    except Exception as e: st.warning(f"Erro (ratings_insights) '{app_name}': {e}"); return "N/A"

@st.cache_data(show_spinner="Gerando insights sobre temas...")
def analyze_topics_insights(gemini_model, app_name, topics_data_json_str):
    prompt = f"App '{app_name}', temas e sentimentos:\n{topics_data_json_str}\nForneça parágrafo (máx 70 palavras) com 1-2 temas +/- significativos e implicações."
    try: return gemini_model.generate_content(prompt).text.strip()
    except Exception as e: st.warning(f"Erro (topics_insights) '{app_name}': {e}"); return "N/A"

@st.cache_data(show_spinner="Gerando insights sobre sentimento comparativo...")
def analyze_comparative_sentiment_insights(gemini_model, sentiment_comparison_str):
    prompt = f"Sentimento geral comparativo:\n{sentiment_comparison_str}\nForneça parágrafo (máx 70 palavras) comparando perfis de sentimento e implicações."
    try: return gemini_model.generate_content(prompt).text.strip()
    except Exception as e: st.warning(f"Erro (comp_sent_insights): {e}"); return "N/A"

@st.cache_data(show_spinner="Gerando análise qualitativa geral...")
def generate_competitive_qualitative_analysis(gemini_model, all_apps_sentiment_topic_analyses, my_app_name):
    input_text_for_gemini = "\n\n".join([
        f"App: {analysis['app_name']}\n- Sentimento: {json.dumps(analysis['sentiment_summary'])}\n- Temas: {json.dumps(analysis['top_topics'])}"
        for analysis in all_apps_sentiment_topic_analyses if "error" not in analysis.get("sentiment_summary", {})
    ])
    if not input_text_for_gemini: return "Dados insuficientes para análise qualitativa geral."
    prompt = f"""Analista de mercado. Baseado nos dados de sentimento/temas (incluindo '{my_app_name}'), forneça análise qualitativa da concorrência em Markdown.
Dados: {input_text_for_gemini}
Estruture:
1.  **Para cada app individualmente:** Pontos Fortes e Fracos.
2.  **Análise Consolidada:**
    * Oportunidades Gerais.
    * Ameaças/Desafios Comuns.
    * **Especulação sobre Tendências Emergentes (máx 2-3 pontos)**.
    * **Sugestão de Posicionamento Estratégico para '{my_app_name}' (1-2 sugestões)**."""
    try: return gemini_model.generate_content(prompt).text.strip()
    except Exception as e: st.error(f"Erro (gen_qual_analysis): {e}"); return f"Erro: {e}"

# --- Funções de Visualização ---
def plot_comparative_sentiment_chart(all_apps_sentiment_data):
    if not all_apps_sentiment_data: st.info("Nenhum dado de sentimento para plotar."); return
    plot_data = {}
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

if 'analysis_complete' not in st.session_state: st.session_state.analysis_complete = False
if 'all_apps_processed_data' not in st.session_state: st.session_state.all_apps_processed_data = []
if 'feature_gap_report' not in st.session_state: st.session_state.feature_gap_report = ""
if 'pain_delight_report' not in st.session_state: st.session_state.pain_delight_report = ""
if 'overall_qualitative_report' not in st.session_state: st.session_state.overall_qualitative_report = ""
if 'my_app_name_for_synthesis_st' not in st.session_state: st.session_state.my_app_name_for_synthesis_st = ""

if analyze_button:
    if not my_app_url_st:
        st.sidebar.warning("Por favor, insira a URL do seu app.")
    elif not model: # Verifica se o modelo Gemini foi inicializado
        st.error("O modelo Gemini não foi inicializado. Verifique a configuração da API Key.")
    else:
        st.session_state.analysis_complete = False
        st.session_state.all_apps_processed_data = []
        st.session_state.feature_gap_report = ""
        st.session_state.pain_delight_report = ""
        st.session_state.overall_qualitative_report = ""
        st.session_state.my_app_name_for_synthesis_st = ""

        urls_to_process_st = [{"url": my_app_url_st, "role": "Meu App"}]
        for i, comp_url in enumerate(competitor_urls_st): urls_to_process_st.append({"url": comp_url, "role": f"Concorrente {i+1}"})

        progress_bar = st.sidebar.progress(0)
        progress_status_text = st.sidebar.empty()
        total_tasks_per_app = 4 # fetch, sentiment/topics, features, pain/delight
        total_synthesis_tasks = 4 # comparative sentiment insights, feature gap, pain/delight comparison, overall qualitative
        total_steps = len(urls_to_process_st) * total_tasks_per_app + total_synthesis_tasks
        
        with st.spinner("Iniciando coleta e análise... Isso pode levar alguns minutos."):
            processed_data_list = []
            step_counter = 0

            for i, app_info in enumerate(urls_to_process_st):
                display_role_url = f"{app_info['role']} ({app_info['url'][:50]}...)"
                progress_status_text.info(f"Processando: {display_role_url}")
                
                app_id_st = get_app_id_from_url(app_info['url'])
                current_app_proc_data = {"display_name": f"{app_info['role']}: {app_id_st}", "review_texts_str": "",
                                         "review_scores": [], "sentiment_topic_analysis": {},
                                         "feature_details": {}, "pain_delight_points": {}}
                if app_id_st:
                    step_counter += 1; progress_bar.progress(step_counter / total_steps); progress_status_text.info(f"Coletando reviews de {app_info['role']}...")
                    texts, scores, name = fetch_play_store_reviews_and_name(app_id_st, count=MAX_REVIEWS_TO_PROCESS)
                    display_name_st = f"{app_info['role']}: {name}" if app_info['role'] != name else name
                    current_app_proc_data["display_name"] = display_name_st
                    current_app_proc_data["review_scores"] = scores
                    if app_info['role'] == "Meu App": st.session_state.my_app_name_for_synthesis_st = display_name_st
                    
                    if texts:
                        reviews_str_st = "\n".join(texts); current_app_proc_data["review_texts_str"] = reviews_str_st
                        step_counter += 1; progress_bar.progress(step_counter / total_steps); progress_status_text.info(f"Analisando sentimento/temas de {display_name_st}...")
                        current_app_proc_data["sentiment_topic_analysis"] = analyze_single_app_reviews(model, reviews_str_st, display_name_st)
                        step_counter += 1; progress_bar.progress(step_counter / total_steps); progress_status_text.info(f"Extraindo features de {display_name_st}...")
                        current_app_proc_data["feature_details"] = extract_feature_details_from_reviews(model, display_name_st, reviews_str_st)
                        step_counter += 1; progress_bar.progress(step_counter / total_steps); progress_status_text.info(f"Extraindo dor/encantamento de {display_name_st}...")
                        current_app_proc_data["pain_delight_points"] = extract_pain_delight_points_from_reviews(model, display_name_st, reviews_str_st)
                    else:
                        current_app_proc_data["sentiment_topic_analysis"] = {"app_name": display_name_st, "sentiment_summary": {"no_reviews": 100.0}, "top_topics": []}
                        current_app_proc_data["feature_details"] = {"elogiadas": [], "problematicas": [], "desejadas_ausentes": []}
                        current_app_proc_data["pain_delight_points"] = {"pontos_dor": [], "fatores_encantamento": []}
                processed_data_list.append(current_app_proc_data)
            st.session_state.all_apps_processed_data = processed_data_list

            if processed_data_list:
                if not st.session_state.my_app_name_for_synthesis_st and processed_data_list:
                    st.session_state.my_app_name_for_synthesis_st = processed_data_list[0]["display_name"]
                
                # Geração de insights comparativos e gerais
                # (A chamada para analyze_comparative_sentiment_insights está na seção de display)
                
                step_counter += 1; progress_bar.progress(step_counter / total_steps); progress_status_text.info("Gerando análise de GAPs de funcionalidades...")
                feature_details_list = [{"app_name": d["display_name"], **d["feature_details"]} for d in processed_data_list if d.get("feature_details")]
                if feature_details_list: st.session_state.feature_gap_report = synthesize_feature_gap_analysis(model, feature_details_list, st.session_state.my_app_name_for_synthesis_st)

                step_counter += 1; progress_bar.progress(step_counter / total_steps); progress_status_text.info("Gerando comparativo de dor/encantamento...")
                pain_delight_list = [{"app_name": d["display_name"], **d["pain_delight_points"]} for d in processed_data_list if d.get("pain_delight_points")]
                if pain_delight_list: st.session_state.pain_delight_report = synthesize_pain_delight_comparison(model, pain_delight_list, st.session_state.my_app_name_for_synthesis_st)
                
                step_counter += 1; progress_bar.progress(step_counter / total_steps); progress_status_text.info("Gerando análise qualitativa geral...")
                sent_topic_list = [d["sentiment_topic_analysis"] for d in processed_data_list if d.get("sentiment_topic_analysis") and "no_reviews" not in d["sentiment_topic_analysis"].get("sentiment_summary", {})]
                if sent_topic_list: st.session_state.overall_qualitative_report = generate_competitive_qualitative_analysis(model, sent_topic_list, st.session_state.my_app_name_for_synthesis_st)
            
            st.session_state.analysis_complete = True
            progress_status_text.success("Análise Concluída!")
            progress_bar.progress(1.0)

if st.session_state.analysis_complete and st.session_state.all_apps_processed_data:
    st.header("🏁 Resultados da Análise Competitiva")
    tab_titles = ["📊 Sentimento Geral", "📱 Apps Individuais", "🧩 GAPs de Features", "❤️ Dor vs. Encantamento", "💡 Qualitativa Estratégica"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

    with tab1:
        st.subheader("1. Análise Comparativa de Sentimentos")
        valid_sent_analyses = [d["sentiment_topic_analysis"] for d in st.session_state.all_apps_processed_data if d.get("sentiment_topic_analysis") and "no_reviews" not in d["sentiment_topic_analysis"].get("sentiment_summary",{}) and "error" not in d["sentiment_topic_analysis"].get("sentiment_summary",{})]
        if valid_sent_analyses:
            plot_comparative_sentiment_chart(valid_sent_analyses)
            sent_comp_str = "\n".join([f"- App: {s['app_name']}, Sentimento: Positivo={s['sentiment_summary'].get('positive',0)}%, Neutro={s['sentiment_summary'].get('neutral',0)}%, Negativo={s['sentiment_summary'].get('negative',0)}%" for s in valid_sent_analyses])
            if sent_comp_str and model: st.markdown(f"**Análise Profissional (Sentimento Comparativo):**\n{analyze_comparative_sentiment_insights(model, sent_comp_str)}")
        else: st.info("Não há dados de sentimento suficientes.")

    with tab2:
        st.subheader("2. Análises Individuais Detalhadas")
        for app_data_st in st.session_state.all_apps_processed_data:
            display_name_st_tab = app_data_st["display_name"]
            # Determina se o expander deve iniciar aberto (para "Meu App")
            is_my_app = st.session_state.my_app_name_for_synthesis_st == display_name_st_tab

            with st.expander(f"Ver detalhes para: {display_name_st_tab}", expanded=is_my_app):
                st.markdown(f"**2a. Distribuição de Notas - {display_name_st_tab}**")
                if app_data_st["review_scores"]:
                    ratings_dist_str_st, _ = plot_ratings_distribution_chart(display_name_st_tab, app_data_st["review_scores"])
                    if ratings_dist_str_st and "Nenhuma nota válida" not in ratings_dist_str_st and model: st.markdown(f"**Análise (Notas):**\n{analyze_ratings_insights(model, display_name_st_tab, ratings_dist_str_st)}")
                else: st.caption(f"Sem dados de notas para {display_name_st_tab}.")
                st.markdown("---")
                st.markdown(f"**2b. Principais Temas - {display_name_st_tab}**")
                s_t_analysis_st = app_data_st["sentiment_topic_analysis"]
                if s_t_analysis_st and "no_reviews" not in s_t_analysis_st.get("sentiment_summary",{}) and "error" not in s_t_analysis_st.get("sentiment_summary",{}) and s_t_analysis_st.get("top_topics"):
                    plot_topics_chart_for_app(s_t_analysis_st)
                    topics_json_str_st = json.dumps(s_t_analysis_st.get("top_topics", []), ensure_ascii=False, indent=2)
                    if model: st.markdown(f"**Análise (Temas):**\n{analyze_topics_insights(model, display_name_st_tab, topics_json_str_st)}")
                    with st.popover("Dados brutos dos temas"): st.json(s_t_analysis_st.get("top_topics", []))
                else: st.caption(f"Sem dados de temas ou reviews para {display_name_st_tab}.")
                st.markdown("---")
                st.markdown(f"**2c. Detalhes de Funcionalidades - {display_name_st_tab}**")
                if app_data_st.get("feature_details") and (app_data_st["feature_details"].get("elogiadas") or app_data_st["feature_details"].get("problematicas") or app_data_st["feature_details"].get("desejadas_ausentes")):
                    st.json(app_data_st["feature_details"])
                else: st.caption(f"Não foram extraídos detalhes de funcionalidades para {display_name_st_tab}.")
                st.markdown("---")
                st.markdown(f"**2d. Pontos de Dor e Encantamento - {display_name_st_tab}**")
                if app_data_st.get("pain_delight_points") and (app_data_st["pain_delight_points"].get("pontos_dor") or app_data_st["pain_delight_points"].get("fatores_encantamento")):
                    st.json(app_data_st["pain_delight_points"])
                else: st.caption(f"Não foram extraídos pontos de dor/encantamento para {display_name_st_tab}.")

    with tab3:
        st.subheader("3. Análise de Funcionalidades e GAPs Competitivos")
        if st.session_state.feature_gap_report: st.markdown(st.session_state.feature_gap_report)
        else: st.info("Relatório de GAPs de funcionalidades não gerado ou dados insuficientes.")

    with tab4:
        st.subheader("4. Comparativo de Pontos de Dor e Fatores de Encantamento")
        if st.session_state.pain_delight_report: st.markdown(st.session_state.pain_delight_report)
        else: st.info("Relatório comparativo de dor/encantamento não gerado ou dados insuficientes.")
            
    with tab5:
        st.subheader("5. Análise Qualitativa Geral da Concorrência e Estratégia")
        if st.session_state.overall_qualitative_report: st.markdown(st.session_state.overall_qualitative_report)
        else: st.info("Análise qualitativa geral não gerada ou dados insuficientes.")
elif not analyze_button:
    st.info("⬅️ Insira as URLs dos apps na barra lateral e clique em 'Analisar Aplicativos' para começar.")

st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido com IA Avançada ✨")
