"""Human-readable labels shared by charts and prediction forms."""
DISPLAY_LABELS = {
    "age": "Age", "balance": "Yearly balance (EUR)", "day": "Contact day",
    "campaign": "Campaign contacts", "pdays": "Days since prior contact",
    "previous": "Prior contacts", "job": "Occupation", "marital": "Marital status",
    "education": "Education", "default": "Credit in default", "housing": "Housing loan",
    "loan": "Personal loan", "contact": "Contact method", "month": "Contact month",
    "poutcome": "Previous outcome", "deposit": "Subscribed", "deposit_flag": "Subscribed",
    "was_contacted_previously": "Previously contacted", "contact_band": "Campaign contacts",
}
FIELD_HELP = {
    "balance": "Average yearly account balance in euros, not income or the amount invested.",
    "day": "Day of the planned contact, from 1 to 31.",
    "campaign": "Include the planned contact in this campaign's count.",
    "pdays": "Use -1 if there was no contact in a previous campaign.",
    "previous": "Number of contacts before the current campaign; use 0 if none.",
    "poutcome": "Use Unknown if the previous campaign result was not recorded.",
    "contact": "The method planned for this contact; Unknown means it was not recorded.",
    "job": "Unknown is valid when occupation was not recorded.",
    "education": "Highest recorded education level; Unknown is a valid category.",
}


def display_category(value):
    return {"admin.": "Administration", "unknown": "Unknown / not recorded"}.get(value, str(value).replace("-", " ").capitalize())


def threshold_explanation(threshold):
    return (f"A score of {threshold:.1%} or above is classified as a likely subscription. "
            "This cutoff maximized F1 on out-of-fold training predictions, balancing precision and recall. "
            "Lowering a cutoff generally finds more potential subscribers but also flags more non-subscribers. "
            "Higher recall can reduce missed opportunities; the right operating point also depends on contact capacity and cost.")
