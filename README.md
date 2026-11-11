# App AI Review: competitive intelligence from Google Play reviews

> **PT-BR:** Análise de concorrência a partir dos reviews da Play Store. Você informa o seu app e até 3 concorrentes; a IA compara sentimento, temas, funcionalidades, dores e encantamentos, e monta uma matriz SWOT.

App reviews are the largest free source of voice-of-customer data a product or marketing team has, but nobody reads thousands of them across four apps. This tool collects them and turns them into a comparison you can use in positioning, roadmap and messaging discussions.

## What it does

- Collects up to **500 of the newest reviews** per app from Google Play (`google-play-scraper`), for your app and up to 3 competitors.
- Uses Gemini to analyse each app:
  - sentiment split and the main topics, with positive, neutral and negative mentions;
  - features users praise, features that cause problems and features they ask for;
  - pain points and delight factors, with example reviews.
- Synthesises across apps: feature gaps, pain/delight comparison, market opportunities and threats, a **SWOT** for your app and a profile of the category's audience.
- Plots sentiment by app, topics by sentiment and the rating distribution.

## Stack

Python · Streamlit · Gemini API · google-play-scraper · pandas · matplotlib / seaborn

## Run it yourself

```bash
pip install -r requirements.txt
# .streamlit/secrets.toml → GOOGLE_API_KEY = "..."
streamlit run "app review.py"
```

Paste the Play Store URLs (`https://play.google.com/store/apps/details?id=...`) in the sidebar and click **Analisar Aplicativos**.

## What I'd do next

- Update the Gemini model name (the code points to a 1.5 model).
- Export the synthesis as a one-page brief for sales and product.
