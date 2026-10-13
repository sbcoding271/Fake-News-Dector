# Fake News Detector — Streamlit UI

This is a UI version of the Fake News Detector ML project.

## Model
- TF-IDF Vectorizer
- Logistic Regression
- Fake/Real classification

## Files
- `app.py` — Streamlit UI
- `model.jb` — trained Logistic Regression model
- `vectorizer.jb` — trained TF-IDF vectorizer
- `requirements.txt` — required packages

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.
