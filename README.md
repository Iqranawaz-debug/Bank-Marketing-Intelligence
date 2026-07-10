# Bank Marketing — Streamlit App

Interactive dashboard + live prediction tool built on top of the EDA/modeling project.

## What's inside
- **📊 Overview** — key metrics and headline findings at a glance
- **🔍 Explore the Data** — filterable, interactive charts (age/balance distributions, job×marital heatmap, monthly volume vs. conversion, correlation heatmap)
- **🤖 Predict a Customer** — fill in a customer profile, get a live subscription probability from the trained XGBoost model, with a gauge chart and top drivers
- **🧠 Model Performance** — full model comparison table, feature importances, and methodology notes

## Files
- `app.py` — the Streamlit app
- `train_model.py` — reproduces the cleaning/preprocessing/training pipeline and saves `model.pkl`, `columns.pkl`, `metrics.pkl`, `bank_clean.csv`, `feature_importances.csv` (already generated — you don't need to rerun this unless you change the data or pipeline)
- `bank.csv` — raw dataset
- `bank_clean.csv` — cleaned dataset used by the dashboard tabs
- `model.pkl`, `columns.pkl`, `metrics.pkl`, `feature_importances.csv` — pre-trained model artifacts

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

It will open automatically at `http://localhost:8501`.

## If you change the data or pipeline
Re-run the training script first to regenerate the model artifacts:
```bash
python train_model.py
```
Then restart the app.

## Deploying it publicly (for LinkedIn / portfolio)
The easiest free option is **Streamlit Community Cloud**:
1. Push this folder to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub, and point it at `app.py`
3. You'll get a public URL you can drop straight into a LinkedIn post
