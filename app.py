import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2563eb"
SUCCESS = "#16a34a"
DANGER = "#dc2626"
BG_CARD = "#0f172a"

st.markdown(f"""
<style>
    .stApp {{ background-color: #0b1120; }}
    .metric-card {{
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
    }}
    .metric-card h2 {{ color: white; margin: 0; font-size: 30px; }}
    .metric-card p {{ color: #94a3b8; margin: 4px 0 0 0; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;}}
    h1, h2, h3 {{ color: #f1f5f9 !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #1e293b; border-radius: 8px 8px 0 0; padding: 10px 18px; color: #cbd5e1;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {PRIMARY} !important; color: white !important; }}
    .prediction-box {{
        border-radius: 16px; padding: 30px; text-align: center; margin-top: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Data & model loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("bank_clean.csv")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    metrics = joblib.load("metrics.pkl")
    return model, columns, metrics

@st.cache_data
def load_importances():
    return pd.read_csv("feature_importances.csv", index_col=0)

df = load_data()
model, model_columns, metrics = load_model()
importances = load_importances()

df_display = df.copy()
df_display["deposit_flag"] = df_display["deposit"].map({"yes": 1, "no": 0})

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 Bank Marketing")
    st.markdown("Intelligence Dashboard")
    st.markdown("---")
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown(f"**Subscription rate:** {df_display['deposit_flag'].mean()*100:.1f}%")
    st.markdown(f"**Best model:** XGBoost")
    st.markdown(f"**ROC-AUC:** {metrics['ROC-AUC']:.3f}")
    st.markdown("---")
    st.caption("Built from an EDA + predictive modeling project on the Bank Marketing dataset (Kaggle, janiobachmann). Duration was excluded from modeling to avoid target leakage.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("# Bank Marketing Campaign Intelligence")
st.markdown("##### Explore the data, understand what drives subscriptions, and score new customers in real time.")
st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Explore the Data", "🤖 Predict a Customer", "🧠 Model Performance"])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{len(df):,}</h2><p>Total Customers</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2>{df_display["deposit_flag"].mean()*100:.1f}%</h2><p>Subscription Rate</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h2>{metrics["ROC-AUC"]:.3f}</h2><p>Model ROC-AUC</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h2>{metrics["Accuracy"]*100:.1f}%</h2><p>Model Accuracy</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("#### Conversion rate by job")
        job_rate = df_display.groupby("job")["deposit_flag"].mean().sort_values(ascending=True) * 100
        fig = px.bar(job_rate, orientation="h", labels={"value": "Subscription Rate (%)", "job": ""},
                     color=job_rate.values, color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Previous outcome impact")
        po = df_display.groupby("poutcome")["deposit_flag"].mean().sort_values(ascending=False) * 100
        fig = px.pie(values=po.values, names=po.index, hole=0.5,
                      color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Key takeaways")
    k1, k2, k3 = st.columns(3)
    k1.info("**Prior success is gold.** Customers who said 'yes' before convert at 91%+ this time.")
    k2.success("**Students & retirees convert best.** Both segments beat the overall base rate by 25+ points.")
    k3.warning("**Age is U-shaped, not linear.** Both very young and older customers out-convert the middle.")

# ============================================================================
# TAB 2: EXPLORE
# ============================================================================
with tab2:
    st.markdown("#### Filter the data")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        job_filter = st.multiselect("Job", sorted(df["job"].unique()), default=[])
    with fc2:
        marital_filter = st.multiselect("Marital status", sorted(df["marital"].unique()), default=[])
    with fc3:
        age_range = st.slider("Age range", int(df.age.min()), int(df.age.max()),
                               (int(df.age.min()), int(df.age.max())))

    filtered = df_display.copy()
    if job_filter:
        filtered = filtered[filtered["job"].isin(job_filter)]
    if marital_filter:
        filtered = filtered[filtered["marital"].isin(marital_filter)]
    filtered = filtered[(filtered.age >= age_range[0]) & (filtered.age <= age_range[1])]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Age distribution by outcome")
        fig = px.histogram(filtered, x="age", color="deposit", barmode="overlay", nbins=40,
                            color_discrete_map={"yes": SUCCESS, "no": DANGER}, opacity=0.7)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Balance distribution by outcome")
        fig = px.box(filtered, x="deposit", y="balance", color="deposit",
                     color_discrete_map={"yes": SUCCESS, "no": DANGER})
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Job × Marital status heatmap (subscription rate)")
        pivot = pd.crosstab(filtered["job"], filtered["marital"], values=filtered["deposit_flag"], aggfunc="mean")
        fig = px.imshow(pivot, color_continuous_scale="YlGnBu", aspect="auto", text_auto=".2f")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Monthly campaign volume vs. conversion")
        month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        m = filtered.groupby("month").agg(volume=("deposit_flag", "count"), rate=("deposit_flag", "mean")).reindex(month_order).dropna()
        fig = go.Figure()
        fig.add_bar(x=m.index, y=m.volume, name="Volume", marker_color="#334155")
        fig.add_scatter(x=m.index, y=m.rate * m.volume.max(), name="Conversion rate (scaled)",
                          yaxis="y", line=dict(color=SUCCESS, width=3))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Correlation heatmap (numeric features)")
    num_cols = ["age", "balance", "day", "campaign", "pdays", "previous", "was_contacted_previously", "deposit_flag"]
    corr = filtered[num_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: PREDICT
# ============================================================================
with tab3:
    st.markdown("#### Score a hypothetical customer")
    st.caption("Fill in the details below (no call duration needed — this predicts *before* the call happens).")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 95, 40)
            job = st.selectbox("Job", sorted(df["job"].unique()))
            marital = st.selectbox("Marital status", sorted(df["marital"].unique()))
            education = st.selectbox("Education", sorted(df["education"].unique()))
        with c2:
            balance = st.number_input("Account balance (€)", value=1000, step=100)
            housing = st.selectbox("Has housing loan?", ["no", "yes"])
            loan = st.selectbox("Has personal loan?", ["no", "yes"])
            default = st.selectbox("Has credit in default?", ["no", "yes"])
        with c3:
            contact = st.selectbox("Contact method", sorted(df["contact"].unique()))
            month = st.selectbox("Contact month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], index=4)
            campaign = st.slider("Contacts this campaign", 1, 15, 2)
            poutcome = st.selectbox("Previous campaign outcome", sorted(df["poutcome"].unique()))

        c4, c5 = st.columns(2)
        with c4:
            previous = st.slider("Number of previous contacts", 0, 20, 0)
        with c5:
            pdays = st.slider("Days since last contact (-1 = never)", -1, 400, -1)

        submitted = st.form_submit_button("🔮 Predict Subscription Likelihood", use_container_width=True)

    if submitted:
        was_contacted_previously = 0 if pdays == -1 else 1
        day = 15  # neutral default, day-of-month has negligible standalone effect

        input_dict = {
            "age": age, "balance": balance, "day": day, "campaign": campaign,
            "pdays": pdays, "previous": previous, "was_contacted_previously": was_contacted_previously,
            "job": job, "marital": marital, "education": education, "default": default,
            "housing": housing, "loan": loan, "contact": contact, "month": month, "poutcome": poutcome,
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, columns=["job","marital","education","default",
                                                            "housing","loan","contact","month","poutcome"])
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        proba = model.predict_proba(input_encoded)[0][1]
        pred = "Likely to Subscribe ✅" if proba >= 0.5 else "Unlikely to Subscribe ❌"
        color = SUCCESS if proba >= 0.5 else DANGER

        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(135deg, {color}22 0%, #0f172a 100%); border: 2px solid {color};">
            <h1 style="color:{color}; margin:0;">{proba*100:.1f}%</h1>
            <p style="color:#e2e8f0; font-size:18px; margin-top:6px;">{pred}</p>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={'suffix': "%", 'font': {'color': '#e2e8f0'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': '#1e293b'},
                    {'range': [40, 70], 'color': '#334155'},
                    {'range': [70, 100], 'color': '#475569'},
                ],
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=280,
                           margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### What's driving this prediction")
        top_feats = importances.head(6)
        st.dataframe(top_feats.rename(columns={"importance": "Global Importance"}), use_container_width=True)
        st.caption("These are the model's globally most important features — not a per-prediction explanation, but a guide to what generally moves the needle.")

# ============================================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================================
with tab4:
    st.markdown("#### Final model: XGBoost Classifier")
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (label, key) in zip([m1,m2,m3,m4,m5],
                                   [("Accuracy","Accuracy"), ("Precision","Precision"),
                                    ("Recall","Recall"), ("F1 Score","F1"), ("ROC-AUC","ROC-AUC")]):
        col.markdown(f'<div class="metric-card"><h2>{metrics[key]:.3f}</h2><p>{label}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### Top 15 Feature Importances")
        top15 = importances.head(15).sort_values("importance")
        fig = px.bar(top15, x="importance", y=top15.index, orientation="h",
                     color="importance", color_continuous_scale="Viridis")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Why XGBoost was chosen")
        st.markdown("""
        Three models were trained and compared on the same held-out test set:

        | Model | Accuracy | Precision | Recall | ROC-AUC |
        |---|---|---|---|---|
        | Logistic Regression | 0.692 | 0.709 | 0.593 | 0.758 |
        | Random Forest | 0.728 | 0.761 | 0.621 | 0.785 |
        | **XGBoost** | **0.741** | **0.769** | **0.647** | **0.796** |

        XGBoost was selected as the production model because it had the
        highest ROC-AUC — the most relevant metric for ranking a call list
        by likelihood, rather than a hard yes/no cutoff.

        **Note on `duration`:** call duration was deliberately excluded from
        every model. It's only known *after* a call ends, so including it
        would leak information from the future and make the model unusable
        for its actual purpose — deciding who to call *before* dialing.
        """)

    st.markdown("---")
    st.caption("Model trained on 80% of 11,162 records, evaluated on a held-out 20% test set with stratified sampling. See the full analysis notebook for the complete EDA, outlier treatment, and modeling methodology.")
