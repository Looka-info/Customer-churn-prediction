"""
Customer Churn Prediction — Streamlit Dashboard Application
A clean, editorial web application for exploring customer churn patterns,
evaluating classification models, running SQL analytics, and generating churn risk predictions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from churn_prediction.config import get_config
from dashboard.analytics_service import (
    ANALYTICAL_QUERIES,
    execute_readonly_query,
    is_database_available,
)
from dashboard.components import (
    inject_custom_css,
    render_empty_state,
    render_metric_card,
    render_status_badge,
)
from dashboard.data_service import (
    check_artifacts_status,
    get_clean_dataset_cached,
    get_dataset_summary,
    load_dataset_cached,
)
from dashboard.model_service import (
    get_best_model_summary,
    get_customer_presets,
    get_feature_schema,
    get_model_comparison_report,
    run_customer_prediction,
)

# 1. Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject clean editorial styling
inject_custom_css()


def sidebar_navigation() -> str:
    """Render compact, descriptive sidebar navigation and compact status area."""
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <h2 style="margin: 0; color: #0f172a; font-size: 1.3rem; font-weight: 700;">Customer Churn</h2>
            <div style="color: #64748b; font-size: 0.85rem;">Interactive ML System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If programmatic navigation was requested via session state
    nav_options = [
        "Overview",
        "Predict Churn",
        "Explore Data",
        "Model Performance",
        "Reports & SQL",
    ]

    default_index = 0
    if "nav_selection" in st.session_state and st.session_state["nav_selection"] in nav_options:
        default_index = nav_options.index(st.session_state["nav_selection"])

    page = st.sidebar.radio(
        "Navigation Menu",
        options=nav_options,
        index=default_index,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Compact, non-intrusive system status
    artifacts = check_artifacts_status()
    all_ready = all(item["exists"] for item in artifacts.values())

    with st.sidebar.expander("System Status", expanded=False):
        if all_ready:
            st.markdown("<span class='status-ok'>✓ All Artifacts Ready</span>", unsafe_allow_html=True)
            st.caption("Dataset, models, and database are loaded.")
        else:
            st.markdown("<span class='status-missing'>⚠ Missing Artifacts</span>", unsafe_allow_html=True)
            st.caption("Run: `python train.py`")

        st.caption("IBM Telco Churn Benchmark | v1.0.0")

    return page


# ==============================================================================
# Page 1: Overview
# ==============================================================================
def render_overview_page():
    st.title("Customer Churn Prediction")
    st.markdown("Explore customer churn patterns and estimate churn risk using a trained machine learning model.")

    st.markdown("")

    # Prominent Primary Call-to-Action
    col_cta, _ = st.columns([2, 3])
    with col_cta:
        if st.button("🔮 Predict Customer Churn", type="primary", use_container_width=True):
            st.session_state["nav_selection"] = "Predict Churn"
            st.rerun()

    st.markdown("---")

    # Four useful dataset metrics from real data
    summary = get_dataset_summary()
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Customers", value=f"{summary['total_records']:,}")
        with col2:
            st.metric(label="Overall Churn Rate", value=f"{summary['churn_rate_pct']:.1f}%")
        with col3:
            st.metric(label="Churned Customers", value=f"{summary['churned_count']:,}")
        with col4:
            st.metric(label="Retained Customers", value=f"{summary['retained_count']:,}")

    st.markdown("")

    # Meaningful visualization: balance between churned and retained customers
    if summary:
        st.markdown("#### Customer Retention vs. Churn Balance")
        fig, ax = plt.subplots(figsize=(8, 1.8))
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#ffffff")

        categories = ["Retained (No Churn)", "Churned"]
        values = [summary["retained_count"], summary["churned_count"]]
        percentages = [
            (summary["retained_count"] / summary["total_records"]) * 100,
            (summary["churned_count"] / summary["total_records"]) * 100,
        ]
        colors = ["#15803d", "#e11d48"]  # Restrained forest green and muted coral/red

        bars = ax.barh(categories, values, color=colors, height=0.45)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.annotate(
                f"{int(width):,} ({percentages[i]:.1f}%)",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color="#0f172a",
                fontweight="600",
                fontsize=10,
            )

        ax.set_xlim(0, summary["total_records"] * 1.15)
        ax.set_xlabel("Number of Customer Accounts", color="#475569", fontsize=9)
        ax.tick_params(colors="#0f172a", labelsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.3, color="#cbd5e1")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#cbd5e1")

        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Three simple navigation cards
    st.markdown("#### Where to go next:")
    ncol1, ncol2, ncol3 = st.columns(3)
    with ncol1:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-title">Predict Churn</div>
                <div style="font-size: 0.95rem; color: #334155; margin-bottom: 12px;">
                    Enter customer details to receive an instant, probabilistic churn risk estimate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Predictor", use_container_width=True, key="btn_to_predict"):
            st.session_state["nav_selection"] = "Predict Churn"
            st.rerun()

    with ncol2:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-title">Explore Data</div>
                <div style="font-size: 0.95rem; color: #334155; margin-bottom: 12px;">
                    Discover churn relationships across contract types, payment methods, and tenure.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Customer Data", use_container_width=True, key="btn_to_explore"):
            st.session_state["nav_selection"] = "Explore Data"
            st.rerun()

    with ncol3:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-title">Model Performance</div>
                <div style="font-size: 0.95rem; color: #334155; margin-bottom: 12px;">
                    Compare cross-validation and test metrics across all three candidate models.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Compare Models", use_container_width=True, key="btn_to_models"):
            st.session_state["nav_selection"] = "Model Performance"
            st.rerun()

    # Expandable technical health check
    st.markdown("")
    with st.expander("🛠️ System Status & Pipeline Artifacts", expanded=False):
        artifacts = check_artifacts_status()
        artifact_rows = []
        for name, item in artifacts.items():
            artifact_rows.append(
                {
                    "Artifact": name,
                    "Status": "Ready" if item["exists"] else "Missing",
                    "Size": item["size"],
                    "Last Modified": item["modified"],
                    "Path": item["path"],
                }
            )
        st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# Page 2: Predict Churn
# ==============================================================================
def render_predict_page():
    st.title("Predict Customer Churn")
    st.markdown(
        "Enter a customer's service and billing information to estimate their churn risk. "
        "This is a model estimate, not a guarantee that a customer will leave."
    )

    cfg = get_config()
    if not cfg.best_model_path.exists():
        render_empty_state(
            "Trained Model (models/best_model.joblib)",
            "python train.py",
            "The model must be trained and saved before interactive predictions can be made.",
        )
        return

    presets = get_customer_presets()
    schema = get_feature_schema()

    # Step 3: Example customer profiles (clearly labeled as illustrative)
    example_names = [
        "Custom Input",
        "Example: Month-to-Month Fiber Profile",
        "Example: Two-Year Contract Family Profile",
        "Example: One-Year Contract Moderate Profile",
    ]

    selected_example = st.selectbox(
        "Choose an illustrative example profile or enter custom details:",
        options=example_names,
        index=0,
        help="These example profiles are illustrative scenarios for demonstration, not real customers with known outcomes.",
    )

    if selected_example != "Custom Input" and selected_example in presets:
        defaults = presets[selected_example]
    else:
        defaults = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 75.0,
            "TotalCharges": 900.0,
        }

    st.markdown("")

    # Form with 3 clear sections
    with st.form("churn_prediction_form"):
        st.markdown("### 1. Customer Profile")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            gender = st.selectbox("Gender", options=schema["gender"], index=schema["gender"].index(defaults["gender"]))
            partner = st.selectbox("Has Partner", options=schema["Partner"], index=schema["Partner"].index(defaults["Partner"]))
        with col_p2:
            senior_val = defaults["SeniorCitizen"]
            senior = st.selectbox(
                "Senior Citizen",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                index=[0, 1].index(senior_val),
            )
            dependents = st.selectbox("Has Dependents", options=schema["Dependents"], index=schema["Dependents"].index(defaults["Dependents"]))
        with col_p3:
            tenure = st.slider(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=int(defaults["tenure"]),
                help="Number of months customer has been subscribed.",
            )

        st.markdown("### 2. Services")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            phone_service = st.selectbox("Phone Service", options=schema["PhoneService"], index=schema["PhoneService"].index(defaults["PhoneService"]))
            multiple_lines = st.selectbox("Multiple Lines", options=schema["MultipleLines"], index=schema["MultipleLines"].index(defaults["MultipleLines"]))
            internet_service = st.selectbox("Internet Service", options=schema["InternetService"], index=schema["InternetService"].index(defaults["InternetService"]))
        with col_s2:
            online_security = st.selectbox("Online Security", options=schema["OnlineSecurity"], index=schema["OnlineSecurity"].index(defaults["OnlineSecurity"]))
            online_backup = st.selectbox("Online Backup", options=schema["OnlineBackup"], index=schema["OnlineBackup"].index(defaults["OnlineBackup"]))
            device_protection = st.selectbox("Device Protection", options=schema["DeviceProtection"], index=schema["DeviceProtection"].index(defaults["DeviceProtection"]))
        with col_s3:
            tech_support = st.selectbox("Tech Support", options=schema["TechSupport"], index=schema["TechSupport"].index(defaults["TechSupport"]))
            streaming_tv = st.selectbox("Streaming TV", options=schema["StreamingTV"], index=schema["StreamingTV"].index(defaults["StreamingTV"]))
            streaming_movies = st.selectbox("Streaming Movies", options=schema["StreamingMovies"], index=schema["StreamingMovies"].index(defaults["StreamingMovies"]))

        st.markdown("### 3. Billing & Contract")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            contract = st.selectbox("Contract Type", options=schema["Contract"], index=schema["Contract"].index(defaults["Contract"]))
            paperless_billing = st.selectbox("Paperless Billing", options=schema["PaperlessBilling"], index=schema["PaperlessBilling"].index(defaults["PaperlessBilling"]))
        with col_b2:
            payment_method = st.selectbox("Payment Method", options=schema["PaymentMethod"], index=schema["PaymentMethod"].index(defaults["PaymentMethod"]))
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=15.0,
                max_value=150.0,
                value=float(defaults["MonthlyCharges"]),
                step=0.5,
            )
        with col_b3:
            calc_default = 0.0 if tenure == 0 else float(defaults["TotalCharges"])
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=calc_default,
                step=5.0,
                help="Lifetime total billed to customer. Automatically 0.0 if tenure is 0.",
            )

        # Advanced Settings: Collapsed by default
        with st.expander("⚙️ Advanced Settings (Decision Threshold)", expanded=False):
            st.caption("Default threshold is 0.50. Lower thresholds increase recall (flagging more potential churners) at the cost of precision.")
            threshold = st.slider("Classification Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

        st.markdown("")
        submitted = st.form_submit_button("Estimate Churn Risk", type="primary", use_container_width=True)

    # Step 4: Present results clearly
    if submitted:
        if tenure == 0 and total_charges > 0.0:
            total_charges = 0.0

        customer_payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            result = run_customer_prediction(customer_payload, threshold=threshold)
            prob = float(result["churn_probability"])
            pred = int(result["predicted_churn"])
            tier = str(result["risk_tier"])

            st.markdown("---")
            st.markdown("### Estimation Results")

            # 1. Estimated Churn Risk (Low, Medium, or High with text & accessible styling)
            badge_class = "risk-badge-low" if tier == "Low Risk" else ("risk-badge-medium" if tier == "Medium Risk" else "risk-badge-high")

            rcol1, rcol2 = st.columns([1, 1])
            with rcol1:
                st.markdown(
                    f"""
                    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;">
                        <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 8px;">
                            Estimated Churn Risk
                        </div>
                        <span class="{badge_class}">{tier}</span>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #0f172a; margin-top: 14px;">
                            Prediction: <strong>{"Likely to Churn" if pred == 1 else "Likely to Remain"}</strong>
                        </div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                            Decision threshold used: {threshold:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rcol2:
                st.markdown(
                    f"""
                    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;">
                        <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 8px;">
                            Estimated Churn Probability
                        </div>
                        <div style="font-size: 2rem; font-weight: 800; color: #0f172a;">
                            {prob * 100:.1f}%
                        </div>
                        <div style="font-size: 0.85rem; color: #475569; margin-top: 6px;">
                            Recommended operational action:
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #15803d; margin-top: 2px;">
                            {result["recommended_action"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("")
            st.info(
                "**What this means**: This prediction is a model estimate based on patterns observed in historical customer data. "
                "It indicates how frequently customers with this profile churned historically, but it does not guarantee customer departure or retention success."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==============================================================================
# Page 3: Explore Data
# ==============================================================================
def render_explore_data_page():
    st.title("Explore Customer Data")
    st.markdown("Understand churn relationships across customer tenure, contract types, and payment methods.")

    clean_df = get_clean_dataset_cached()
    if clean_df is None:
        render_empty_state(
            "Raw Dataset (data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)",
            "python train.py",
            "The raw dataset must be downloaded before exploration is possible.",
        )
        return

    # Four core charts
    st.markdown("### Key Churn Relationships")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Chart 1: Churn by Contract Type
        rates_contract = clean_df.groupby("Contract")["Churn"].mean() * 100
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        fig1.patch.set_facecolor("#fafafa")
        ax1.set_facecolor("#ffffff")
        bars1 = ax1.bar(rates_contract.index, rates_contract.values, color="#15803d", width=0.45)
        for b in bars1:
            ax1.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", color="#0f172a", fontweight="600", fontsize=9)
        ax1.set_ylabel("Churn Rate (%)", color="#475569", fontsize=9)
        ax1.set_title("Churn Rate by Contract Type", color="#0f172a", fontweight="600", fontsize=11, pad=10)
        ax1.tick_params(colors="#0f172a", labelsize=9)
        ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#cbd5e1")
        for spine in ["top", "right"]:
            ax1.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax1.spines[spine].set_color("#cbd5e1")
        st.pyplot(fig1)
        plt.close()

        # Chart 2: Churn by Tenure Cohort
        clean_df_tenure = clean_df.copy()
        clean_df_tenure["Tenure_Cohort"] = pd.cut(
            clean_df_tenure["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12 Mos", "13-24 Mos", "25-48 Mos", "49-72 Mos"],
        )
        rates_tenure = clean_df_tenure.groupby("Tenure_Cohort", observed=False)["Churn"].mean() * 100
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor("#fafafa")
        ax2.set_facecolor("#ffffff")
        bars2 = ax2.bar(rates_tenure.index.astype(str), rates_tenure.values, color="#3b82f6", width=0.45)
        for b in bars2:
            ax2.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", color="#0f172a", fontweight="600", fontsize=9)
        ax2.set_ylabel("Churn Rate (%)", color="#475569", fontsize=9)
        ax2.set_title("Churn Rate by Customer Tenure", color="#0f172a", fontweight="600", fontsize=11, pad=10)
        ax2.tick_params(colors="#0f172a", labelsize=9)
        ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#cbd5e1")
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax2.spines[spine].set_color("#cbd5e1")
        st.pyplot(fig2)
        plt.close()

    with chart_col2:
        # Chart 3: Churn by Payment Method
        rates_pm = clean_df.groupby("PaymentMethod")["Churn"].mean() * 100
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        fig3.patch.set_facecolor("#fafafa")
        ax3.set_facecolor("#ffffff")
        bars3 = ax3.bar(rates_pm.index, rates_pm.values, color="#d97706", width=0.45)
        for b in bars3:
            ax3.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", color="#0f172a", fontweight="600", fontsize=9)
        ax3.set_ylabel("Churn Rate (%)", color="#475569", fontsize=9)
        ax3.set_title("Churn Rate by Payment Method", color="#0f172a", fontweight="600", fontsize=11, pad=10)
        ax3.tick_params(colors="#0f172a", labelsize=8)
        plt.xticks(rotation=15, ha="right")
        ax3.grid(axis="y", linestyle="--", alpha=0.3, color="#cbd5e1")
        for spine in ["top", "right"]:
            ax3.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax3.spines[spine].set_color("#cbd5e1")
        st.pyplot(fig3)
        plt.close()

        # Chart 4: Monthly Charges by Churn Status
        plot_df = clean_df.copy()
        plot_df["Status"] = plot_df["Churn"].map({1: "Churned", 0: "Retained"})
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        fig4.patch.set_facecolor("#fafafa")
        ax4.set_facecolor("#ffffff")
        sns.boxplot(data=plot_df, x="Status", y="MonthlyCharges", ax=ax4, palette=["#15803d", "#e11d48"], width=0.4)
        ax4.set_ylabel("Monthly Charges ($)", color="#475569", fontsize=9)
        ax4.set_xlabel("")
        ax4.set_title("Monthly Charges by Churn Status", color="#0f172a", fontweight="600", fontsize=11, pad=10)
        ax4.tick_params(colors="#0f172a", labelsize=9)
        ax4.grid(axis="y", linestyle="--", alpha=0.3, color="#cbd5e1")
        for spine in ["top", "right"]:
            ax4.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax4.spines[spine].set_color("#cbd5e1")
        st.pyplot(fig4)
        plt.close()

    st.markdown("---")

    # Limited, controlled customer records preview (customerID hidden by default)
    st.markdown("### Customer Records Sample")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        f_churn = st.selectbox("Filter Status:", ["All Customers", "Churned Only", "Retained Only"])
    with fcol2:
        f_contract = st.selectbox("Filter Contract:", ["All Contracts"] + sorted(list(clean_df["Contract"].unique())))
    with fcol3:
        show_id = st.checkbox("Show customerID", value=False)

    df_view = clean_df.copy()
    if f_churn == "Churned Only":
        df_view = df_view[df_view["Churn"] == 1]
    elif f_churn == "Retained Only":
        df_view = df_view[df_view["Churn"] == 0]

    if f_contract != "All Contracts":
        df_view = df_view[df_view["Contract"] == f_contract]

    cols_to_show = list(df_view.columns)
    if not show_id and "customerID" in cols_to_show:
        cols_to_show.remove("customerID")

    st.dataframe(df_view[cols_to_show].head(15), use_container_width=True, hide_index=True)
    st.caption(f"Showing first {min(15, len(df_view))} of {len(df_view):,} matching customer accounts.")

    # Optional deep dive
    with st.expander("Additional Exploratory Charts", expanded=False):
        st.markdown("#### Churn Rate by Internet Service Tier")
        rates_internet = clean_df.groupby("InternetService")["Churn"].mean() * 100
        fig_int, ax_int = plt.subplots(figsize=(6, 2.5))
        fig_int.patch.set_facecolor("#fafafa")
        ax_int.set_facecolor("#ffffff")
        bars_int = ax_int.bar(rates_internet.index, rates_internet.values, color="#15803d", width=0.4)
        for b in bars_int:
            ax_int.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", color="#0f172a", fontweight="600", fontsize=9)
        ax_int.set_ylabel("Churn Rate (%)", color="#475569", fontsize=9)
        ax_int.tick_params(colors="#0f172a", labelsize=9)
        ax_int.grid(axis="y", linestyle="--", alpha=0.3, color="#cbd5e1")
        for spine in ["top", "right"]:
            ax_int.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax_int.spines[spine].set_color("#cbd5e1")
        st.pyplot(fig_int)
        plt.close()


# ==============================================================================
# Page 4: Model Performance
# ==============================================================================
def render_model_performance_page():
    st.title("Model Performance")
    st.markdown("Evaluating Logistic Regression, Random Forest, and Gradient Boosting on cross-validation and held-out test data.")

    comp_df = get_model_comparison_report()
    if comp_df is None:
        render_empty_state(
            "Model Comparison Report (reports/model_comparison.csv)",
            "python train.py",
            "Run the full pipeline to generate the empirical model comparison report.",
        )
        return

    st.markdown(
        """
        <div class="editorial-callout">
            <strong>Model Selection Rationale</strong>: Three classification models were evaluated using 5-fold cross-validation on 5,634 training accounts, 
            and audited on 1,409 held-out test accounts. <strong>Random Forest</strong> was retained as the default model because it achieved higher 
            cross-validation precision (67.39%), minimizing false alarms under marketing retention outreach constraints.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Model Comparison Table")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Selectable single metric comparison chart
    st.markdown("### Compare by Metric")
    metric_cols = [c for c in comp_df.columns if c != "Model"]
    selected_metric = st.selectbox(
        "Choose metric to visualize across candidate models:",
        options=metric_cols,
        index=metric_cols.index("Test_PR_AUC") if "Test_PR_AUC" in metric_cols else 0,
    )

    chart_df = comp_df[["Model", selected_metric]].copy()
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#ffffff")
    bars = ax.bar(chart_df["Model"], chart_df[selected_metric], color="#15803d", width=0.45)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#0f172a",
            fontweight="600",
            fontsize=9,
        )
    ax.set_ylabel(selected_metric, color="#475569", fontsize=9)
    ax.tick_params(colors="#0f172a", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#cbd5e1")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cbd5e1")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Metric definitions made understandable
    st.markdown("### Understanding Evaluation Metrics")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-title">Precision</div>
                <div style="font-size: 0.9rem; color: #334155;">
                    Of the customers flagged as at-risk, what percentage actually churned? Crucial when outreach offers cost money.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-title">Recall</div>
                <div style="font-size: 0.9rem; color: #334155;">
                    Of all customers who actually left, what percentage did the model catch? Crucial when preventing loss is the priority.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol3:
        st.markdown(
            """
            <div class="custom-metric-card">
                <div class="custom-metric-title">PR-AUC (Average Precision)</div>
                <div style="font-size: 0.9rem; color: #334155;">
                    Summary of precision across all recall thresholds on the minority positive class. Uninflated by true negatives.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Held-out test confusion matrix
    st.markdown("### Held-Out Test Confusion Matrix (Random Forest @ 0.50)")
    cfg = get_config()
    cm_path = cfg.figures_dir / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Evaluated on 1,409 held-out test accounts", width=500)
    else:
        st.info("Confusion matrix figure not found. Run pipeline to generate.")

    # Advanced technical details collapsed
    with st.expander("🔬 Statistical Uncertainty & Nadeau-Bengio Test", expanded=False):
        st.markdown(
            r"""
            - **Nadeau–Bengio Corrected Resampled $t$-Test**:
              Standard paired $t$-tests violate independence because training folds overlap by 80%. Using the corrected test across 3-repeated 5-fold CV ($R=15, df=14, \text{ratio}=0.25$):
              - **Random Forest vs. Logistic Regression**: $\bar{d} = +0.0044$, $p = 0.3557$.
              - **Gradient Boosting vs. Logistic Regression**: $\bar{d} = +0.0049$, $p = 0.4285$.
            - **Defensible Finding**: The observed performance differences ($+0.0044$ and $+0.0049$) fall well within fold-to-fold resampling variance ($\pm 0.020$). All three models are statistically comparable.
            """
        )

    with st.expander("📉 Diagnostic Curves (PR & ROC)", expanded=False):
        curves_path = cfg.figures_dir / "pr_roc_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), caption="Precision-Recall and ROC diagnostic curves", use_container_width=True)

    with st.expander("⭐ Feature Importance Breakdown", expanded=False):
        fi_path = cfg.figures_dir / "feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), caption="Permutation feature importance on held-out test set", use_container_width=True)


# ==============================================================================
# Page 5: Reports & SQL (Combined Clean Page)
# ==============================================================================
def render_reports_and_sql_page():
    st.title("Reports & SQL Analytics")
    st.markdown("Relational database insights and technical documentation.")

    tab_sql, tab_reports = st.tabs(["💼 Business Insights (SQL)", "📑 Technical Reports & Governance"])

    with tab_sql:
        if not is_database_available():
            render_empty_state(
                "SQLite Database (data/churn_analysis.db)",
                "python train.py",
                "The SQLite database must be populated before running queries.",
            )
        else:
            queries = ANALYTICAL_QUERIES
            query_titles = [q["title"] for q in queries]
            selected_title = st.selectbox("Select a business question to analyze:", options=query_titles)
            selected_q = next(q for q in queries if q["title"] == selected_title)

            st.markdown(f"**Insight**: {selected_q['description']}")

            try:
                sql_df = execute_readonly_query(selected_q["sql"])
                st.dataframe(sql_df, use_container_width=True, hide_index=True)

                # Chart query result if churn_rate_percent exists
                if "churn_rate_percent" in sql_df.columns:
                    cat_cols = [c for c in sql_df.columns if c not in ["total_customers", "churned_customers", "churn_rate_percent", "avg_monthly_charges", "avg_monthly_bill"]]
                    if cat_cols:
                        cat_col = cat_cols[0]
                        fig, ax = plt.subplots(figsize=(6, 2.5))
                        fig.patch.set_facecolor("#fafafa")
                        ax.set_facecolor("#ffffff")
                        bars = ax.bar(sql_df[cat_col].astype(str), sql_df["churn_rate_percent"], color="#15803d", width=0.45)
                        for b in bars:
                            ax.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", color="#0f172a", fontweight="600", fontsize=9)
                        ax.set_ylabel("Churn Rate (%)", color="#475569", fontsize=9)
                        ax.set_title(f"Churn Rate by {cat_col}", color="#0f172a", fontweight="600", fontsize=11, pad=10)
                        ax.tick_params(colors="#0f172a", labelsize=9)
                        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#cbd5e1")
                        for spine in ["top", "right"]:
                            ax.spines[spine].set_visible(False)
                        for spine in ["left", "bottom"]:
                            ax.spines[spine].set_color("#cbd5e1")
                        st.pyplot(fig)
                        plt.close()

                # View underlying SQL hidden in expander
                with st.expander("View SQL Query", expanded=False):
                    st.code(selected_q["sql"], language="sql")

            except Exception as e:
                st.error(f"SQL execution error: {e}")

    with tab_reports:
        st.markdown("### Project Governance & Methodology")
        st.markdown(
            r"""
            - **Dataset Provenance**: IBM Telco Customer Churn Public Sample Dataset, created for IBM Cognos Analytics.
            - **License**: Code is licensed under [MIT License](https://opensource.org/licenses/MIT). Dataset is property of IBM Corporation.
            - **No Timestamps**: The dataset contains zero timestamps (no sign-up date, invoice dates, or churn dates). This project is strictly a retrospective classification benchmark, not a forward-looking 30-day forecast.
            - **Retention Campaign Utility**:
              $$\Delta \text{Value} = (\text{TP} \times \eta \times V) - (\text{TP} + \text{FP}) \times C_{\text{outreach}}$$
              False negatives cancel out algebraically when comparing against the baseline of doing nothing.
            """
        )

        st.markdown("---")
        st.markdown("### Command-Line Reproduction")
        st.code(
            """
# 1. Run automated test suite (38 unit and integration tests)
python -m pytest

# 2. Run end-to-end pipeline (downloads data, verifies SQL, trains models)
python train.py

# 3. Launch the dashboard
streamlit run app.py
            """.strip(),
            language="bash",
        )


# ==============================================================================
# Main Router
# ==============================================================================
def main():
    selected_page = sidebar_navigation()

    if selected_page == "Overview":
        render_overview_page()
    elif selected_page == "Predict Churn":
        render_predict_page()
    elif selected_page == "Explore Data":
        render_explore_data_page()
    elif selected_page == "Model Performance":
        render_model_performance_page()
    elif selected_page == "Reports & SQL":
        render_reports_and_sql_page()


if __name__ == "__main__":
    main()
