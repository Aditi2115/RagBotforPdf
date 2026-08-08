# RagBot for PDF

A Streamlit app that lets you upload a PDF and ask questions about its contents.

## Live Demo

[Open RagBot for PDF](https://ragbotforpdf-aaeg84apwcdjkxnq62z5em.streamlit.app/)

## Run Locally

Install the dependencies and start Streamlit:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Create a local `.env` file with your Gemini API key:

```dotenv
GEMINI_API_KEY=your-api-key
```

For Streamlit Cloud, add the key under the app's **Settings > Secrets**:

```toml
GEMINI_API_KEY = "your-api-key"
```

Do not commit `.env` or API keys to GitHub.
