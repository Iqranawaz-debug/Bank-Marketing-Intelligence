"""Bank Marketing Intelligence: a saved-model Streamlit data product."""
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.config import CATEGORIES, FEATURES, LABELS, NUMERIC
from src.data_processing import load_data, read_prediction_csv
from src.predict import load_artifacts, local_sensitivity, predict_customers
from src.presentation import DISPLAY_LABELS, FIELD_HELP, display_category, threshold_explanation
from src.visualization import TEAL, insight_text, rate_chart, rate_table, style_chart

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Bank Marketing Intelligence", page_icon="🏦", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 4.5rem; max-width: 1480px;}
[data-testid="stMetric"] {background: white; border: 1px solid #dce5ed; border-radius: 12px; padding: 16px;}
[data-testid="stMetricLabel"] {color: #526680;}
[data-testid="stMetricLabel"] p {white-space: normal;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
h1 {letter-spacing: -1.1px;}
[data-testid="stSidebar"] {border-right: 1px solid #dce5ed;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def cached_data():
    logger.info("Application dataset loaded")
    return load_data()


@st.cache_resource
def cached_model():
    logger.info("Application model startup")
    return load_artifacts()


def chart(figure, height=380):
    st.plotly_chart(style_chart(figure, height), width="stretch")


def metric_row(values):
    for column, (name, value) in zip(st.columns(len(values)), values):
        column.metric(name, value)


def heading(kicker, title, text):
    st.caption(kicker.upper())
    st.title(title)
    st.write(text)


try:
    df = cached_data()
except Exception:
    logger.error("Dataset initialization failed")
    st.error("The project dataset could not be loaded. Restore bank.csv with the expected schema and restart.")
    st.stop()

model, report = None, None
try:
    model, report = cached_model()
except Exception:
    logger.error("Model initialization failed")

PAGES = ["Executive Dashboard", "Exploratory Analysis", "Business Insights", "Customer Prediction", "Batch Prediction", "Model Performance", "Model Explainability", "Model Information", "About Project"]
with st.sidebar:
    st.markdown("## Bank Marketing Intelligence")
    st.caption("Campaign analytics & customer response")
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("DATASET SNAPSHOT")
    st.write(f"**{len(df):,}** customer records")
    st.write(f"**{df.deposit_flag.mean():.1%}** subscribed in this sample")
    if report:
        st.caption(f"Selected model · {report['selected_model']}")
    st.caption("Portfolio demonstration · historical sample")

if report is None:
    st.warning("Prediction artifacts are unavailable or incompatible. Run `python train_model.py` in the project environment, then restart the app. Data exploration remains available.")
    if page in PAGES[3:8]:
        st.stop()

if page == "Executive Dashboard":
    heading("Campaign analytics & prediction", "Bank Marketing Intelligence", "Analyze historical bank campaigns and estimate whether a customer will subscribe to a term deposit before a planned call.")
    metric_row([("Customer records", f"{len(df):,}"), ("Sample subscription", f"{df.deposit_flag.mean():.1%}"), ("Selected model", report["selected_model"] if report else "Unavailable"), ("Test ROC-AUC", f"{report['metrics']['ROC-AUC']:.3f}" if report else "Unavailable")])
    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        chart(rate_chart(df, "job", "Subscription rate by occupation"), 430)
    with right:
        chart(rate_chart(df, "poutcome", "Previous campaign outcomes"), 310)
        st.info("The dotted line is the overall sample subscription rate. Hover for group sizes. Unknown outcomes remain unknown; they do not always mean no prior contact.")
    st.subheader("What this project demonstrates")
    st.write("Exploratory analysis · Business analytics · Machine learning · Model evaluation · Explainability · Interactive prediction")
    st.subheader("Historical findings")
    for line in insight_text(df)[:3]:
        st.write(line)
    st.caption("Observed sample rates are not bank-wide conversion forecasts. Model scores are uncalibrated estimates, not guaranteed outcomes.")

elif page == "Exploratory Analysis":
    heading("Explore / historical sample", "Explore campaign responses", "Filter the original records. All charts and rates below use the selected cohort, without modeling caps.")
    c1, c2, c3 = st.columns(3)
    jobs = c1.multiselect("Occupation", CATEGORIES["job"])
    marital = c2.multiselect("Marital status", CATEGORIES["marital"])
    ages = c3.slider("Age range", int(df.age.min()), int(df.age.max()), (int(df.age.min()), int(df.age.max())))
    cohort = df[df.age.between(*ages)]
    if jobs:
        cohort = cohort[cohort.job.isin(jobs)]
    if marital:
        cohort = cohort[cohort.marital.isin(marital)]
    if cohort.empty:
        st.info("No records match these filters. Widen the selection.")
        st.stop()
    metric_row([("Selected records", f"{len(cohort):,}"), ("Subscription rate", f"{cohort.deposit_flag.mean():.1%}"), ("Median campaign contacts", f"{cohort.campaign.median():.0f}")])
    group = st.selectbox("Analysis focus", ["Dataset overview", "Customer demographics", "Financial characteristics", "Campaign and contact", "Historical outcomes", "Feature relationships"])
    if group == "Dataset overview":
        chart(px.histogram(cohort, x="deposit", color="deposit", labels={**DISPLAY_LABELS, "count": "Customer records"}, title="Subscriptions in the selected cohort", color_discrete_sequence=[TEAL, "#8597ad"]))
        st.caption("This sample is close to balanced. Its response mix may differ from live campaigns.")
        st.dataframe(cohort.drop(columns="deposit_flag").head(200).rename(columns=DISPLAY_LABELS), width="stretch", hide_index=True)
        st.caption("Explorer shows the first 200 matching records. The download contains the full cohort.")
        st.download_button("Download filtered data", cohort.drop(columns="deposit_flag").to_csv(index=False).encode(), "filtered_bank.csv", "text/csv")
    elif group == "Customer demographics":
        a, b = st.columns(2)
        with a:
            chart(px.histogram(cohort, x="age", color="deposit", labels={**DISPLAY_LABELS, "count": "Customer records"}, nbins=35, barmode="overlay", opacity=.7, title="Age distribution by response"))
        with b:
            chart(rate_chart(cohort, "education", "Subscription by education"))
        pivot = pd.crosstab(cohort.job, cohort.marital, values=cohort.deposit_flag, aggfunc="mean")
        pivot.index.name, pivot.columns.name = "Occupation", "Marital status"
        chart(px.imshow(pivot, text_auto=".0%", zmin=0, zmax=1, aspect="auto", color_continuous_scale="Teal", title="Occupation × marital status: observed rate"), 470)
        st.caption("Distributions show counts; the heatmap shows rates. Small subgroups can produce unstable rates; empty cells have no observations.")
    elif group == "Financial characteristics":
        chart(px.box(cohort, x="deposit", y="balance", color="deposit", labels=DISPLAY_LABELS, title="Yearly balance by subscription outcome"))
        feature = st.selectbox("Financial attribute", ["housing", "loan", "default"], format_func=DISPLAY_LABELS.get)
        chart(rate_chart(cohort, feature, f"Subscription by {LABELS[feature].lower()}"))
        st.caption("Extreme balances are retained here. The model caps balance using limits learned only from its training rows.")
    elif group == "Campaign and contact":
        table = rate_table(cohort, "month").set_index("month").reindex(CATEGORIES["month"]).reset_index()
        a, b = st.columns(2)
        with a:
            chart(px.bar(table, x="month", y="Customers", labels={"month": "Contact month", "Customers": "Customer records"}, title="Monthly campaign volume", color_discrete_sequence=[TEAL]))
        with b:
            chart(px.line(table, x="month", y="Rate (%)", labels={"month": "Contact month", "Rate (%)": "Subscription rate (%)"}, markers=True, title="Monthly subscription rate"))
        chart(rate_chart(cohort, "contact", "Subscription by contact method"))
        grouped = cohort.assign(contact_band=pd.cut(cohort.campaign, [0, 1, 3, 5, 10000], labels=["1", "2–3", "4–5", "6+"]))
        chart(rate_chart(grouped, "contact_band", "Subscription by number of campaign contacts"))
        st.caption("Volume and rate have separate axes and units. Month and contact count associations may reflect campaign selection or repeated attempts.")
    elif group == "Historical outcomes":
        chart(rate_chart(cohort, "poutcome", "Subscription by previous campaign outcome"))
        st.dataframe(rate_table(cohort, "poutcome").drop(columns="Rate"), hide_index=True, width="stretch")
        st.caption("Unknown means the previous outcome is unrecorded. Prior contact is independently derived from pdays ≠ -1.")
    else:
        numeric = cohort[NUMERIC + ["deposit_flag"]].copy()
        numeric["was_contacted_previously"] = cohort.pdays.ne(-1).astype(int)
        chart(px.imshow(numeric.rename(columns=DISPLAY_LABELS).corr(), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Pearson correlations"), 540)
        st.caption("Correlation captures linear association, not causality. The pdays=-1 sentinel affects its numeric correlation.")

elif page == "Business Insights":
    heading("Interpret / descriptive evidence", "What the data suggests", "Use these findings to form testable campaign hypotheses. They do not establish which interventions will cause more subscriptions.")
    for feature, title in [("job", "Customer segments"), ("contact", "Contact strategy"), ("poutcome", "Previous campaign response")]:
        a, b = st.columns([1.4, 1])
        with a:
            chart(rate_chart(df, feature, title), 380 if feature == "job" else 280)
        with b:
            table = rate_table(df, feature)
            best = table.iloc[0]
            st.subheader(display_category(best[feature]))
            st.markdown("**Finding**")
            st.write("This group showed the highest observed subscription rate for this attribute in the historical sample.")
            st.markdown("**Evidence**")
            st.write(f"{best['Rate (%)']:.1f}% subscribed, based on {int(best.Customers):,} records; {(best.Rate-df.deposit_flag.mean())*100:+.1f} percentage points versus the sample average.")
            st.markdown("**Business interpretation**")
            interpretations = {"job": "Compare segment response and contact volumes when designing a future campaign experiment.", "contact": "The pattern motivates testing contact channels; it does not prove one channel causes better responses.", "poutcome": "Prior response may help prioritize review of customer history, subject to eligibility and contact preferences."}
            st.write(interpretations[feature])
    summary = "Bank Marketing: descriptive analysis\n\n" + "\n".join(insight_text(df))
    st.download_button("Download analysis summary", summary.encode("utf-8"), "business_insights.txt", "text/plain")

elif page == "Customer Prediction":
    heading("Score / one customer", "Estimate subscription likelihood", "Enter a hypothetical customer and the planned contact details. Call duration is excluded because it is unavailable before the call.")
    st.caption("Start with the first historical example below, or edit the profile. The score is an uncalibrated model estimate.")
    with st.expander("How the decision threshold works"):
        st.write(threshold_explanation(report["threshold"]))
    with st.form("customer_form"):
        columns = st.columns(3)
        values = {}
        defaults = df.iloc[0].to_dict()
        bounds = {"age": (18, 120), "balance": (-1e9, 1e9), "day": (1, 31), "campaign": (1, 10000), "pdays": (-1, 100000), "previous": (0, 10000)}
        groups = [["age", "job", "marital", "education", "balance"], ["default", "housing", "loan", "previous", "pdays"], ["contact", "day", "month", "campaign", "poutcome"]]
        for column, fields, title in zip(columns, groups, ["Customer profile", "Finances & history", "Planned contact"]):
            with column:
                st.subheader(title)
                for feature in fields:
                    if feature in CATEGORIES:
                        choices = CATEGORIES[feature]
                        default = defaults[feature]
                        values[feature] = st.selectbox(LABELS[feature], choices, index=choices.index(default), format_func=display_category, help=FIELD_HELP.get(feature), key=f"input_{feature}")
                    else:
                        low, high = bounds[feature]
                        values[feature] = st.number_input(LABELS[feature], min_value=low, max_value=high, value=float(defaults[feature]) if feature == "balance" else int(defaults[feature]), help=FIELD_HELP.get(feature), key=f"input_{feature}")
        submitted = st.form_submit_button("Predict subscription", type="primary", width="stretch")
    if submitted:
        st.session_state.pop("customer_result", None)
        try:
            row = pd.DataFrame([values])
            result = predict_customers(model, row, report["threshold"])
            st.session_state["customer_result"] = (row, result)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            logger.error("Manual prediction failed")
            st.error("Prediction could not be completed. Check the model environment and retry.")
    if "customer_result" in st.session_state:
        row, result = st.session_state["customer_result"]
        probability = result.subscription_probability.iloc[0]
        st.subheader("Prediction result")
        st.caption("Result reflects the last submitted profile. Submit again after editing fields.")
        outcome = "Likely to subscribe" if result.prediction.iloc[0] == "yes" else "Unlikely to subscribe"
        st.markdown(f"### {outcome}")
        metric_row([("Estimated model probability", f"{probability:.1%}"), ("Decision threshold", f"{report['threshold']:.1%}")])
        st.progress(float(probability))
        st.write("The score is above the decision threshold." if result.prediction.iloc[0] == "yes" else "The score is below the decision threshold.")
        if (row.pdays.iloc[0] == -1 and row.previous.iloc[0] > 0) or (row.pdays.iloc[0] >= 0 and row.previous.iloc[0] == 0):
            st.warning("Previous-contact fields appear inconsistent. Review them before interpreting the score.")
        st.subheader("How this profile changes the score")
        sensitivity = local_sensitivity(model, row, report["reference"]).sort_values("difference")
        sensitivity["feature"] = sensitivity.feature.map(DISPLAY_LABELS)
        sensitivity["difference"] *= 100
        chart(px.bar(sensitivity, x="difference", y="feature", orientation="h", title="Individual feature sensitivity", labels={"difference": "Score difference vs. reference (percentage points)", "feature": "Customer or campaign input"}, color_discrete_sequence=[TEAL]), 430)
        st.caption("Each bar compares this score with the score after replacing one input by its training median/mode. This is sensitivity analysis, not SHAP or a causal explanation. Effects do not add up; replacements may create unrealistic profiles.")
        st.download_button("Download prediction", result.to_csv(index=False).encode(), "customer_prediction.csv", "text/csv")

elif page == "Batch Prediction":
    heading("Score / customer batch", "Turn a CSV into a scored list", "Upload up to 10,000 rows (5 MB). Scoring uses the saved pipeline and never retrains the model.")
    st.write("Download the example → Add your customer rows → Upload, review and download scores.")
    st.download_button("Download example CSV", df[FEATURES].head(5).to_csv(index=False).encode(), "example_customers.csv", "text/csv")
    with st.expander("Required input schema"):
        st.dataframe(pd.DataFrame([{"column": f, "description": LABELS[f], "accepted values": ", ".join(CATEGORIES[f]) if f in CATEGORIES else "Finite number; see example"} for f in FEATURES]), hide_index=True, width="stretch")
        st.write("No blank cells. Use the literal unknown only where listed. Extra columns, including duration, deposit and identifiers, are excluded from scoring and output. Row order is preserved.")
    upload = st.file_uploader("Customer CSV", type=["csv"])
    if upload is not None:
        try:
            batch = read_prediction_csv(upload.getvalue())
            results = predict_customers(model, batch, report["threshold"])
            positives = int(results.prediction.eq("yes").sum())
            metric_row([("Rows scored", f"{len(results):,}"), ("Predicted positive", f"{positives:,}"), ("Predicted negative", f"{len(results)-positives:,}"), ("Positive share", f"{positives/len(results):.1%}")])
            st.caption(f"Classification cutoff: {report['threshold']:.1%}. Positive share is a model output, not an observed conversion rate.")
            with st.expander("Preview validated input"):
                st.dataframe(batch.head(20).rename(columns=DISPLAY_LABELS), hide_index=True, width="stretch")
            st.subheader("Scored customers")
            display_results = results.head(200).rename(columns={**DISPLAY_LABELS, "subscription_probability": "Model score", "prediction": "Predicted subscription", "decision_threshold": "Threshold"})
            display_results["Model score"] = display_results["Model score"].map(lambda value: f"{value:.1%}")
            display_results["Threshold"] = display_results["Threshold"].map(lambda value: f"{value:.1%}")
            st.dataframe(display_results, width="stretch", hide_index=True)
            st.caption("Preview shows up to 200 rows. Download includes every scored row in original order. Uploaded customer data is not written to disk or logged.")
            st.download_button("Download scored CSV", results.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")
        except ValueError as exc:
            logger.warning("Batch validation rejected")
            st.error(str(exc))
        except Exception:
            logger.error("Batch prediction failed")
            st.error("Unable to score this file. Check the model environment and try the example CSV.")

elif page == "Model Performance":
    heading("Evaluate / held-out evidence", f"Selected model: {report['selected_model']}", f"Selected using {report['selection_metric']}. Evaluation below uses {report['test_rows']:,} held-out records.")
    metric_row([(name, f"{report['metrics'][name]:.1%}" if name == "Recall" else f"{report['metrics'][name]:.3f}") for name in ["ROC-AUC", "Average precision", "F1", "Recall"]])
    st.write(f"**Accuracy:** {report['metrics']['Accuracy']:.1%} · **Precision:** {report['metrics']['Precision']:.1%} · **Decision threshold:** {report['threshold']:.1%}")
    st.caption(f"Class metrics use the saved threshold {report['threshold']:.3f}. ROC-AUC and average precision evaluate ranking across thresholds.")
    st.subheader("Model comparison · 0.50 cutoff")
    comparison = pd.DataFrame(report["comparison_at_0_5"])
    comparison.insert(1, "Selected", comparison.Model.eq(report["selected_model"]))
    compact = comparison[["Model", "CV average precision", "ROC-AUC", "Average precision", "Recall", "F1"]].rename(columns={"CV average precision": "CV AP", "Average precision": "Test AP"})
    def highlight_selected(row):
        return ["background-color: #dcefed; font-weight: bold" if row["Model"] == report["selected_model"] else "" for _ in row]
    st.dataframe(compact.style.apply(highlight_selected, axis=1).format({c: "{:.3f}" for c in compact.columns if c != "Model"}), width="stretch", hide_index=True)
    st.caption("AP = average precision. All columns except CV AP are held-out results. Highlighted row is the selected model.")
    winner = comparison[comparison.Model.eq(report["selected_model"])].iloc[0]
    runner = comparison[~comparison.Model.eq(report["selected_model"])].sort_values("CV average precision", ascending=False).iloc[0]
    st.write(f"{report['selected_model']} was selected for CV average precision of {winner['CV average precision']:.3f}, versus {runner['CV average precision']:.3f} for {runner.Model}. This measures ranking quality for subscriptions; it is not a claim of superiority on every metric.")
    with st.expander("Full comparison and CV variability"):
        st.dataframe(comparison, width="stretch", hide_index=True)
    st.caption("Comparison class metrics use a common 0.5 threshold. Selection uses training CV average precision, not the test ranking. CV scores after tuning are not independent estimates of generalization.")
    a, b = st.columns([1, 2])
    curves = report["curves"]
    with a:
        chart(px.imshow(curves["confusion_matrix"], x=["Predicted no", "Predicted yes"], y=["Actual no", "Actual yes"], text_auto=True, color_continuous_scale="Teal", title="Confusion matrix"), 340)
    with b:
        st.subheader("What the errors mean")
        tn, fp = curves["confusion_matrix"][0]
        fn, tp = curves["confusion_matrix"][1]
        st.write(f"The model identified **{tp:,}** of **{tp+fn:,}** subscribers and missed **{fn:,}**. It also flagged **{fp:,}** customers who did not subscribe.")
        st.write(threshold_explanation(report["threshold"]))
    b, c = st.columns(2)
    with b:
        fig = go.Figure(go.Scatter(x=curves["fpr"], y=curves["tpr"], name="Selected model", line_color=TEAL))
        fig.add_scatter(x=[0, 1], y=[0, 1], line_dash="dash", name="Chance")
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", showlegend=False)
        chart(fig, 340)
    with c:
        fig = go.Figure(go.Scatter(x=curves["recall"], y=curves["precision"], line_color=TEAL))
        fig.add_hline(y=report["test_positive_rate"], line_dash="dash")
        fig.update_layout(title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision")
        chart(fig, 340)
    st.subheader("Threshold trade-offs on training predictions")
    thresholds = pd.DataFrame(report["threshold_analysis"])
    chart(px.line(thresholds, x="threshold", y=["Precision", "Recall", "F1"], labels={"threshold": "Decision threshold", "value": "Metric score", "variable": "Metric"}, title="Precision, recall and F1 by threshold"))
    st.caption(report["threshold_method"] + " Hyperparameters were chosen on the same training folds; threshold analysis is not nested validation.")
    st.download_button("Download model comparison", comparison.to_csv(index=False).encode(), "model_comparison.csv", "text/csv")

elif page == "Model Explainability":
    heading("Explain / model behavior", "Which inputs support the model's ranking?", "Permutation importance measures how much held-out average precision drops when a feature is randomly shuffled.")
    st.subheader("Global model behavior")
    importance = pd.DataFrame(report["importance"]).sort_values("mean")
    importance["feature"] = importance.feature.map(DISPLAY_LABELS)
    chart(px.bar(importance, x="mean", y="feature", error_x="std", orientation="h", title="Held-out permutation importance", color_discrete_sequence=[TEAL], labels={"mean": "Drop in average precision", "feature": "Customer or campaign input"}), 560)
    st.info("Higher bars indicate stronger reliance by this fitted model. Error bars show standard deviation across five shuffles, not confidence intervals. Correlated inputs can share importance; negative values can occur through noise.")
    st.write("For business interpretation, a high month or previous-outcome importance means these inputs help distinguish responses in this historical sample. It does not mean changing them will cause a subscription.")
    st.subheader("Individual prediction explanations")
    st.write("Open Customer Prediction and submit a profile to see how its score changes when one input is replaced with a typical training value. These effects describe model sensitivity, not causal effects or additive SHAP contributions.")

elif page == "Model Information":
    heading("Document / reproducible workflow", "The model card", "Predict deposit = yes for a planned campaign contact, using historical customer and campaign information.")
    st.markdown("**Workflow:** raw data → stratified split → train-fitted caps → prior-contact feature → numeric scaling + one-hot encoding → cross-validation and tuning → selected pipeline → saved-model inference.")
    st.write(f"Training: {report['train_rows']:,} rows. Test: {report['test_rows']:,} rows. Split seed: {report['seed']}. Selected algorithm: {report['selected_model']}.")
    st.write("Three existing algorithm families are retained, plus a dummy baseline. A small 3-fold grid tests regularization/depth and unweighted versus mildly weighted classes. No oversampling is used. Average precision prioritizes subscription ranking; ROC-AUC and class metrics provide complementary views.")
    st.write("Balance is capped at training 1st/99th percentiles; campaign at the training 99th percentile. All transforms refit inside each training fold. Unknown remains an explicit category. The API rejects missing or invalid fields. Duration is excluded throughout inference.")
    st.dataframe(pd.DataFrame([{"Feature": f, "Meaning": LABELS[f]} for f in FEATURES]), hide_index=True, width="stretch")
    with st.expander("Chosen parameters, environment and tuning results"):
        st.json({"parameters": report["parameters"], "versions": report["versions"], "python": report["python"], "capping_bounds": report["capping_bounds"], "dataset_sha256": report["dataset_sha256"]})
        st.dataframe(pd.DataFrame(report["tuning"]), width="stretch", hide_index=True)
    st.subheader("Limitations & intended use")
    st.markdown("""- Historical sample, with no verified population sampling design or temporal holdout. Sample conversion is not a live-campaign forecast.
- The original notebook previously examined a similar test split. These results are reproducible holdout diagnostics, not a new external benchmark.
- Scores are uncalibrated. Threshold maximizes training F1 without explicit contact costs or business value.
- Planned month/day/contact/count must be known at scoring time. Historical final campaign counts may differ from what was available at an earlier contact.
- No fairness audit, prospective validation, drift monitoring, authentication, or customer-data retention system is included.
- Use for portfolio demonstration and hypothesis generation; operational adoption needs fresh data and governance review.""")

elif page == "About Project":
    heading("About / portfolio", "Bank Marketing Intelligence", "A reproducible data science project connecting exploratory analysis, model evaluation, and an interactive prediction experience.")
    st.subheader("Dataset")
    st.write("The repository contains bank.csv with 11,162 records, 16 original input columns and the deposit target. The original project attributes it to the Bank Marketing dataset shared by janiobachmann on Kaggle. Confirm redistribution terms before publishing. Duration is excluded from the deployed model.")
    st.subheader("Skills demonstrated")
    st.write("Exploratory analysis · business interpretation · leakage prevention · feature engineering · model comparison · cross-validation · threshold analysis · explainability · input validation · testing · Streamlit application development")
    st.subheader("Technology")
    st.write("Python, pandas, NumPy, scikit-learn, XGBoost, joblib, Plotly, and Streamlit.")
    st.subheader("From analysis to application")
    st.write("Historical data → exploratory analysis → business interpretation → reproducible model evaluation → explainable, interactive scoring.")
    st.caption("This demonstration does not establish causal effects, guarantee customer behavior, or represent a bank's production deployment.")
