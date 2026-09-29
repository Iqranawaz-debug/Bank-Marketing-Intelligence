"""Shared interactive charts and descriptive summaries."""
import plotly.express as px
from src.presentation import DISPLAY_LABELS, display_category

TEAL = "#087f8c"


def rate_table(data, feature):
    table = data.groupby(feature, observed=True).agg(Customers=("deposit_flag", "size"), Subscriptions=("deposit_flag", "sum"), Rate=("deposit_flag", "mean")).reset_index()
    table["Rate (%)"] = table.Rate * 100
    return table.sort_values("Rate (%)", ascending=False)


def rate_chart(data, feature, title=None):
    table = rate_table(data, feature).sort_values("Rate (%)")
    table[feature] = table[feature].astype(str).map(display_category)
    figure = px.bar(table, x="Rate (%)", y=feature, orientation="h", hover_data=["Customers", "Subscriptions"], title=title, labels={feature: DISPLAY_LABELS.get(feature, feature), "Rate (%)": "Subscription rate (%)"}, color_discrete_sequence=[TEAL])
    figure.add_vline(x=data.deposit_flag.mean() * 100, line_dash="dot", line_color="#bd7633")
    figure.update_xaxes(range=[0, 100])
    return figure


def style_chart(figure, height=380):
    if figure.layout.title.text is None:
        figure.update_layout(title_text="")
    figure.update_layout(height=height, margin=dict(l=15, r=15, t=55, b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial", color="#182b49", size=12), title_font_size=17)
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    return figure


def insight_text(data):
    lines = []
    for feature, label in [("job", "occupation"), ("contact", "contact method"), ("poutcome", "previous outcome")]:
        row = rate_table(data, feature).iloc[0]
        lines.append(f"Highest observed rate by {label}: {row[feature]} — {row['Rate (%)']:.1f}% ({int(row.Subscriptions):,} subscriptions / {int(row.Customers):,} records).")
    lines.append("These are descriptive associations in this sample, not causal effects or population forecasts.")
    return lines
