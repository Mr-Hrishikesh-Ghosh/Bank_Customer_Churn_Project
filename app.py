import streamlit as st
import pandas as pd
import joblib
import os
import shap
import numpy as np

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ==================================================
# PDF REPORT FUNCTION
# ==================================================

def create_pdf_report(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card_value,
    is_active_member_value,
    estimated_salary,
    churn_probability,
    churn_status,
    risk_category,
    estimated_ltv,
    retention_cost,
    expected_retention_value,
    expected_net_retention_value,
    retention_priority,
    retention_recommendation,
    final_model_name
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14
    )

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Bank Customer Churn Prediction & Retention System",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Customer Prediction and Retention Analysis Report",
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Model Information",
            heading_style
        )
    )

    model_data = [
        ["Final ML Model", str(final_model_name)],
        ["Prediction Type", "Customer Churn Prediction"],
        ["Analysis Type", "Cost-Sensitive Retention Analysis"]
    ]

    model_table = Table(
        model_data,
        colWidths=[180, 320]
    )

    model_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(
        model_table
    )

    # --------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Customer Information",
            heading_style
        )
    )

    customer_data = [
        ["Credit Score", str(credit_score)],
        ["Geography", geography],
        ["Gender", gender],
        ["Age", str(age)],
        ["Tenure", f"{tenure} years"],
        ["Account Balance", f"₹{balance:,.2f}"],
        ["Number of Products", str(num_of_products)],
        [
            "Has Credit Card",
            "Yes" if has_cr_card_value == 1 else "No"
        ],
        [
            "Active Member",
            "Yes" if is_active_member_value == 1 else "No"
        ],
        ["Estimated Salary", f"₹{estimated_salary:,.2f}"]
    ]

    customer_table = Table(
        customer_data,
        colWidths=[180, 320]
    )

    customer_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(
        customer_table
    )

    # --------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Churn Prediction",
            heading_style
        )
    )

    prediction_data = [
        [
            "Churn Probability",
            f"{churn_probability:.2%}"
        ],
        [
            "Predicted Status",
            churn_status
        ],
        [
            "Risk Category",
            risk_category
        ]
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[180, 320]
    )

    prediction_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(
        prediction_table
    )

    # --------------------------------------------------
    # CUSTOMER VALUE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Customer Value & Retention Economics",
            heading_style
        )
    )

    financial_data = [
        [
            "Estimated Customer LTV",
            f"₹{estimated_ltv:,.2f}"
        ],
        [
            "Retention Intervention Cost",
            f"₹{retention_cost:,.2f}"
        ],
        [
            "Expected Retention Value",
            f"₹{expected_retention_value:,.2f}"
        ],
        [
            "Expected Net Retention Value",
            f"₹{expected_net_retention_value:,.2f}"
        ]
    ]

    financial_table = Table(
        financial_data,
        colWidths=[220, 280]
    )

    financial_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(
        financial_table
    )

    # --------------------------------------------------
    # RETENTION DECISION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Retention Decision",
            heading_style
        )
    )

    retention_data = [
        [
            "Retention Priority",
            retention_priority
        ],
        [
            "Recommended Action",
            Paragraph(
                retention_recommendation,
                normal_style
            )
        ]
    ]

    retention_table = Table(
        retention_data,
        colWidths=[180, 320]
    )

    retention_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(
        retention_table
    )

    # --------------------------------------------------
    # REPORT NOTE
    # --------------------------------------------------

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Report Note",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The estimated LTV represents a customer value proxy "
            "based on the project's predefined business assumptions. "
            "Retention cost and success probability are scenario-based "
            "assumptions and should be replaced with actual business "
            "data when available.",
            normal_style
        )
    )

    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    document.build(
        story
    )

    buffer.seek(0)

    return buffer


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #ffffff;
    }

    .recommendation {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# MODEL PATHS
# ==================================================

MODEL_PATH = "saved_model/final_model_pipeline.pkl"

MODEL_NAME_PATH = "saved_model/final_model_name.pkl"

ASSUMPTIONS_PATH = "saved_model/business_assumptions.pkl"


# ==================================================
# CHECK MODEL FILES
# ==================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model file not found: {MODEL_PATH}"
    )

    st.stop()


if not os.path.exists(MODEL_NAME_PATH):

    st.error(
        f"Model name file not found: {MODEL_NAME_PATH}"
    )

    st.stop()


if not os.path.exists(ASSUMPTIONS_PATH):

    st.error(
        f"Business assumptions file not found: {ASSUMPTIONS_PATH}"
    )

    st.stop()


# ==================================================
# LOAD MODEL
# ==================================================

final_model_pipeline = joblib.load(
    MODEL_PATH
)

final_model_name = joblib.load(
    MODEL_NAME_PATH
)

business_assumptions = joblib.load(
    ASSUMPTIONS_PATH
)


# ==================================================
# TITLE
# ==================================================

st.markdown(
    '<div class="main-title">'
    '🏦 Bank Customer Churn Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered churn prediction, customer value estimation, '
    'and cost-sensitive retention decision support.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# PROJECT OVERVIEW
# ==================================================

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)


with overview_col1:

    st.metric(
        "ML Model",
        str(final_model_name).replace(
            " Tuned",
            ""
        )
    )


with overview_col2:

    st.metric(
        "Dataset Size",
        "10,000 Customers"
    )


with overview_col3:

    st.metric(
        "Prediction",
        "Churn Risk"
    )


with overview_col4:

    st.metric(
        "Retention",
        "LTV + Cost"
    )


st.divider()


# ==================================================
# PROJECT WORKFLOW
# ==================================================

with st.expander(
    "🔄 View System Workflow"
):

    st.markdown(
        """
        ### Bank Customer Churn & Retention Workflow

        **1. Customer Input**  
        Customer demographic, financial and banking information.

        ↓

        **2. Feature Engineering**  
        Balance-to-salary ratio, product engagement,
        tenure group, inactive-member flag and age group.

        ↓

        **3. ML Churn Prediction**  
        The trained final machine-learning model estimates
        the probability that the customer will churn.

        ↓

        **4. Risk Classification**  
        The customer is classified as Low, Medium or High Risk.

        ↓

        **5. Customer Lifetime Value**  
        An estimated customer value proxy is calculated.

        ↓

        **6. Cost-Sensitive Retention Analysis**  
        Retention cost and expected retention value are calculated.

        ↓

        **7. Retention Decision**  
        The system generates a retention priority and
        recommended action.

        ↓

        **8. Explainable AI**  
        SHAP identifies the features influencing the prediction.

        ↓

        **9. Customer Report**  
        Results can be downloaded as CSV or PDF.
        """
    )


# ==================================================
# METHODOLOGY NOTE
# ==================================================

st.info(
    "This system combines machine-learning-based churn "
    "prediction with estimated customer lifetime value "
    "and scenario-based retention economics to support "
    "cost-sensitive customer retention decisions."
)


# ==================================================
# MODEL INFORMATION
# ==================================================

st.info(
    f"🤖 Prediction Model: **{final_model_name}**"
)


# ==================================================
# CUSTOMER INPUT SECTION
# ==================================================

st.header(
    "👤 Customer Information"
)


col1, col2, col3 = st.columns(3)


with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

    geography = st.selectbox(
        "Geography",
        [
            "France",
            "Germany",
            "Spain"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )


with col2:

    tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    num_of_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )


with col3:

    has_cr_card = st.selectbox(
        "Has Credit Card",
        [
            "Yes",
            "No"
        ]
    )

    is_active_member = st.selectbox(
        "Active Member",
        [
            "Yes",
            "No"
        ]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )


# ==================================================
# PREDICTION BUTTON
# ==================================================

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


# ==================================================
# PREDICTION SECTION
# ==================================================

if predict_button:

    # --------------------------------------------------
    # BASIC VALUE CONVERSION
    # --------------------------------------------------

    has_cr_card_value = (
        1
        if has_cr_card == "Yes"
        else 0
    )

    is_active_member_value = (
        1
        if is_active_member == "Yes"
        else 0
    )


    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    balance_salary_ratio = (
        balance /
        (estimated_salary + 1)
    )

    product_engagement_score = (
        num_of_products *
        (1 + is_active_member_value)
    )


    # --------------------------------------------------
    # TENURE BUCKET
    # --------------------------------------------------

    if tenure <= 2:

        tenure_bucket = "0-2"

    elif tenure <= 5:

        tenure_bucket = "3-5"

    elif tenure <= 8:

        tenure_bucket = "6-8"

    else:

        tenure_bucket = "9-10"


    # --------------------------------------------------
    # INACTIVE FLAG
    # --------------------------------------------------

    inactive_flag = (
        1
        if is_active_member_value == 0
        else 0
    )


    # --------------------------------------------------
    # AGE GROUP
    # --------------------------------------------------

    if age <= 25:

        age_group = "18-25"

    elif age <= 35:

        age_group = "26-35"

    elif age <= 45:

        age_group = "36-45"

    elif age <= 55:

        age_group = "46-55"

    else:

        age_group = "56+"


    # ==================================================
    # CREATE CUSTOMER DATAFRAME
    # ==================================================

    customer_data = pd.DataFrame({

        "CreditScore": [
            credit_score
        ],

        "Geography": [
            geography
        ],

        "Gender": [
            gender
        ],

        "Age": [
            age
        ],

        "Tenure": [
            tenure
        ],

        "Balance": [
            balance
        ],

        "NumOfProducts": [
            num_of_products
        ],

        "HasCrCard": [
            has_cr_card_value
        ],

        "IsActiveMember": [
            is_active_member_value
        ],

        "EstimatedSalary": [
            estimated_salary
        ],

        "BalanceSalaryRatio": [
            balance_salary_ratio
        ],

        "ProductEngagementScore": [
            product_engagement_score
        ],

        "TenureBucket": [
            tenure_bucket
        ],

        "InactiveFlag": [
            inactive_flag
        ],

        "AgeGroup": [
            age_group
        ]
    })


    # ==================================================
    # MODEL PREDICTION
    # ==================================================

    churn_probability = (
        final_model_pipeline
        .predict_proba(
            customer_data
        )[:, 1][0]
    )

    prediction = (
        final_model_pipeline
        .predict(
            customer_data
        )[0]
    )


    # --------------------------------------------------
    # CHURN STATUS
    # --------------------------------------------------

    churn_status = (
        "Churned"
        if prediction == 1
        else "Stayed"
    )


    # --------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------

    if churn_probability >= 0.70:

        risk_category = "High Risk"

    elif churn_probability >= 0.40:

        risk_category = "Medium Risk"

    else:

        risk_category = "Low Risk"


    # ==================================================
    # PREDICTION RESULTS
    # ==================================================

    st.divider()

    st.header(
        "📊 Prediction Results"
    )


    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2%}"
        )


    with result_col2:

        st.metric(
            "Predicted Status",
            churn_status
        )


    with result_col3:

        st.metric(
            "Risk Category",
            risk_category
        )


    # ==================================================
    # CUSTOMER LTV CALCULATION
    # ==================================================

    balance_value = (
        business_assumptions[
            "balance_value_rate"
        ] * balance
    )

    salary_value = (
        business_assumptions[
            "salary_value_rate"
        ] * estimated_salary
    )

    product_value = (
        business_assumptions[
            "product_value"
        ] * num_of_products
    )

    activity_value = (
        business_assumptions[
            "activity_value"
        ] * is_active_member_value
    )


    annual_customer_value = (
        balance_value
        + salary_value
        + product_value
        + activity_value
    )


    remaining_tenure = max(
        0,
        10 - tenure
    )


    estimated_ltv = (
        annual_customer_value
        * remaining_tenure
    )


    # ==================================================
    # LTV DISPLAY
    # ==================================================

    st.divider()

    st.header(
        "💰 Customer Lifetime Value"
    )


    st.metric(
        "Estimated Customer LTV",
        f"₹{estimated_ltv:,.2f}"
    )


    # ==================================================
    # RETENTION COST
    # ==================================================

    if risk_category == "High Risk":

        retention_cost = (
            business_assumptions[
                "high_risk_retention_cost"
            ]
        )

        retention_success_probability = (
            business_assumptions[
                "high_risk_success_probability"
            ]
        )


    elif risk_category == "Medium Risk":

        retention_cost = (
            business_assumptions[
                "medium_risk_retention_cost"
            ]
        )

        retention_success_probability = (
            business_assumptions[
                "medium_risk_success_probability"
            ]
        )


    else:

        retention_cost = (
            business_assumptions[
                "low_risk_retention_cost"
            ]
        )

        retention_success_probability = (
            business_assumptions[
                "low_risk_success_probability"
            ]
        )


    # ==================================================
    # EXPECTED RETENTION VALUE
    # ==================================================

    expected_retention_value = (
        churn_probability
        * estimated_ltv
        * retention_success_probability
    )


    expected_net_retention_value = (
        expected_retention_value
        - retention_cost
    )


    # ==================================================
    # RETENTION PRIORITY
    # ==================================================

    if expected_net_retention_value <= 0:

        retention_priority = "Low Priority"

    elif risk_category == "High Risk":

        retention_priority = "Priority Retention"

    elif risk_category == "Medium Risk":

        retention_priority = "Targeted Retention"

    else:

        retention_priority = "Monitor"


    # ==================================================
    # RETENTION RECOMMENDATION
    # ==================================================

    if retention_priority == "Priority Retention":

        retention_recommendation = (
            "Initiate personalized retention "
            "intervention and relationship-manager "
            "follow-up."
        )

    elif retention_priority == "Targeted Retention":

        retention_recommendation = (
            "Provide a targeted retention offer "
            "or personalized communication."
        )

    elif retention_priority == "Monitor":

        retention_recommendation = (
            "Monitor customer activity and maintain "
            "regular engagement."
        )

    else:

        retention_recommendation = (
            "Avoid costly intervention; use low-cost "
            "automated communication if necessary."
        )


    # ==================================================
    # RETENTION DECISION
    # ==================================================

    st.divider()

    st.header(
        "🎯 Retention Decision"
    )


    retention_col1, retention_col2 = (
        st.columns(2)
    )


    with retention_col1:

        st.metric(
            "Retention Priority",
            retention_priority
        )


    with retention_col2:

        st.metric(
            "Risk Level",
            risk_category
        )


    st.info(
        f"💡 **Recommended Action:** "
        f"{retention_recommendation}"
    )


    # ==================================================
    # RETENTION VALUE DETAILS
    # ==================================================

    retention_value_col1, retention_value_col2 = (
        st.columns(2)
    )


    with retention_value_col1:

        st.metric(
            "Retention Cost",
            f"₹{retention_cost:,.2f}"
        )


    with retention_value_col2:

        st.metric(
            "Expected Retention Value",
            f"₹{expected_retention_value:,.2f}"
        )


    st.metric(
        "Expected Net Retention Value",
        f"₹{expected_net_retention_value:,.2f}"
    )


    # ==================================================
    # CUSTOMER SUMMARY
    # ==================================================

    st.divider()

    st.header(
        "👤 Customer Summary"
    )


    summary_col1, summary_col2, summary_col3 = (
        st.columns(3)
    )


    with summary_col1:

        st.write(
            f"**Geography:** {geography}"
        )

        st.write(
            f"**Gender:** {gender}"
        )

        st.write(
            f"**Age:** {age}"
        )

        st.write(
            f"**Tenure:** {tenure} years"
        )


    with summary_col2:

        st.write(
            f"**Credit Score:** {credit_score}"
        )

        st.write(
            f"**Account Balance:** "
            f"₹{balance:,.2f}"
        )

        st.write(
            f"**Products:** {num_of_products}"
        )

        st.write(
            f"**Active Member:** "
            f"{is_active_member}"
        )


    with summary_col3:

        st.write(
            f"**Estimated LTV:** "
            f"₹{estimated_ltv:,.2f}"
        )

        st.write(
            f"**Churn Probability:** "
            f"{churn_probability:.2%}"
        )

        st.write(
            f"**Expected Net Value:** "
            f"₹{expected_net_retention_value:,.2f}"
        )

        st.write(
            f"**Retention Priority:** "
            f"{retention_priority}"
        )


    # ==================================================
    # SHAP EXPLANATION
    # ==================================================

    st.divider()

    st.header(
        "🔎 Why Did the Model Make This Prediction?"
    )


    try:

        final_trained_model = (
            final_model_pipeline
            .named_steps["model"]
        )

        final_preprocessor = (
            final_model_pipeline
            .named_steps["preprocessor"]
        )


        # --------------------------------------------------
        # TRANSFORM CUSTOMER DATA
        # --------------------------------------------------

        customer_transformed = (
            final_preprocessor.transform(
                customer_data
            )
        )


        feature_names = (
            final_preprocessor
            .get_feature_names_out()
        )


        # --------------------------------------------------
        # CREATE SHAP EXPLAINER
        # --------------------------------------------------

        explainer = shap.TreeExplainer(
            final_trained_model
        )


        shap_values = (
            explainer.shap_values(
                customer_transformed
            )
        )


        # --------------------------------------------------
        # HANDLE BINARY CLASSIFICATION
        # --------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            customer_shap_values = (
                shap_values[1][0]
            )

        else:

            customer_shap_values = (
                shap_values[0]
            )


        # --------------------------------------------------
        # CREATE SHAP DATAFRAME
        # --------------------------------------------------

        shap_df = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Value": customer_shap_values

        })


        # --------------------------------------------------
        # ABSOLUTE IMPACT
        # --------------------------------------------------

        shap_df["Impact"] = (
            shap_df["SHAP Value"].abs()
        )


        # --------------------------------------------------
        # HUMAN-READABLE FEATURE NAMES
        # --------------------------------------------------

        feature_name_mapping = {

            "num__CreditScore":
                "Credit Score",

            "num__Age":
                "Age",

            "num__Tenure":
                "Tenure",

            "num__Balance":
                "Account Balance",

            "num__NumOfProducts":
                "Number of Products",

            "num__HasCrCard":
                "Credit Card Ownership",

            "num__IsActiveMember":
                "Active Membership",

            "num__EstimatedSalary":
                "Estimated Salary",

            "num__BalanceSalaryRatio":
                "Balance-to-Salary Ratio",

            "num__ProductEngagementScore":
                "Product Engagement Score",

            "num__InactiveFlag":
                "Inactive Member Flag",

            "cat__Geography_France":
                "Geography: France",

            "cat__Geography_Germany":
                "Geography: Germany",

            "cat__Geography_Spain":
                "Geography: Spain",

            "cat__Gender_Female":
                "Gender: Female",

            "cat__Gender_Male":
                "Gender: Male",

            "cat__TenureBucket_0-2":
                "Tenure Group: 0-2 Years",

            "cat__TenureBucket_3-5":
                "Tenure Group: 3-5 Years",

            "cat__TenureBucket_6-8":
                "Tenure Group: 6-8 Years",

            "cat__TenureBucket_9-10":
                "Tenure Group: 9-10 Years",

            "cat__AgeGroup_18-25":
                "Age Group: 18-25",

            "cat__AgeGroup_26-35":
                "Age Group: 26-35",

            "cat__AgeGroup_36-45":
                "Age Group: 36-45",

            "cat__AgeGroup_46-55":
                "Age Group: 46-55",

            "cat__AgeGroup_56+":
                "Age Group: 56+"
        }


        shap_df["Readable Feature"] = (
            shap_df["Feature"]
            .map(
                feature_name_mapping
            )
            .fillna(
                shap_df["Feature"]
            )
        )


        # --------------------------------------------------
        # SORT BY ABSOLUTE IMPACT
        # --------------------------------------------------

        shap_df = shap_df.sort_values(
            "Impact",
            ascending=False
        )


        # --------------------------------------------------
        # TOP 10 FEATURES
        # --------------------------------------------------

        top_shap = (
            shap_df
            .head(10)
            .copy()
        )


        # --------------------------------------------------
        # RISK-INCREASING FACTORS
        # --------------------------------------------------

        risk_increasing = top_shap[
            top_shap["SHAP Value"] > 0
        ].copy()


        # --------------------------------------------------
        # RISK-REDUCING FACTORS
        # --------------------------------------------------

        risk_reducing = top_shap[
            top_shap["SHAP Value"] < 0
        ].copy()


        # --------------------------------------------------
        # DISPLAY HUMAN-READABLE EXPLANATION
        # --------------------------------------------------

        explanation_col1, explanation_col2 = (
            st.columns(2)
        )


        with explanation_col1:

            st.subheader(
                "🔴 Factors Increasing Churn Risk"
            )


            if len(
                risk_increasing
            ) == 0:

                st.success(
                    "No major risk-increasing factors "
                    "were identified among the top features."
                )

            else:

                for _, row in (
                    risk_increasing.iterrows()
                ):

                    st.write(
                        f"🔴 **{row['Readable Feature']}**"
                    )


        with explanation_col2:

            st.subheader(
                "🟢 Factors Reducing Churn Risk"
            )


            if len(
                risk_reducing
            ) == 0:

                st.info(
                    "No major risk-reducing factors "
                    "were identified among the top features."
                )

            else:

                for _, row in (
                    risk_reducing.iterrows()
                ):

                    st.write(
                        f"🟢 **{row['Readable Feature']}**"
                    )


        # --------------------------------------------------
        # SHAP IMPACT TABLE
        # --------------------------------------------------

        st.subheader(
            "📊 Feature Impact Details"
        )


        display_shap = top_shap[
            [
                "Readable Feature",
                "SHAP Value"
            ]
        ].copy()


        display_shap.columns = [
            "Feature",
            "SHAP Impact"
        ]


        st.dataframe(
            display_shap,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------
        # VISUAL SHAP CHART
        # --------------------------------------------------

        st.subheader(
            "📈 Top Factors Influencing Churn Prediction"
        )


        chart_df = top_shap.copy()


        chart_df = chart_df.sort_values(
            "SHAP Value"
        )


        st.bar_chart(
            chart_df.set_index(
                "Readable Feature"
            )["SHAP Value"]
        )


        st.caption(
            "Positive SHAP values push the prediction "
            "toward churn, while negative SHAP values "
            "push the prediction toward staying with "
            "the bank."
        )


    except Exception as e:

        st.warning(
            f"SHAP explanation could not be generated: {e}"
        )


    # ==================================================
    # DOWNLOAD CUSTOMER REPORT
    # ==================================================

    st.divider()

    st.header(
        "📄 Customer Prediction Report"
    )


    report_data = {

        "Customer Information": [

            "Credit Score",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Account Balance",
            "Number of Products",
            "Has Credit Card",
            "Active Member",
            "Estimated Salary"

        ],

        "Value": [

            credit_score,
            geography,
            gender,
            age,
            tenure,
            f"₹{balance:,.2f}",
            num_of_products,

            "Yes"
            if has_cr_card_value == 1
            else "No",

            "Yes"
            if is_active_member_value == 1
            else "No",

            f"₹{estimated_salary:,.2f}"

        ]
    }


    prediction_report = pd.DataFrame(
        report_data
    )


    # --------------------------------------------------
    # ADD PREDICTION RESULTS
    # --------------------------------------------------

    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Churn Probability",

        f"{churn_probability:.2%}"

    ]


    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Predicted Status",

        churn_status

    ]


    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Risk Category",

        risk_category

    ]


    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Estimated Customer LTV",

        f"₹{estimated_ltv:,.2f}"

    ]


    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Retention Cost",

        f"₹{retention_cost:,.2f}"

    ]


    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Expected Retention Value",

        f"₹{expected_retention_value:,.2f}"

    ]


    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Expected Net Retention Value",

        f"₹{expected_net_retention_value:,.2f}"

    ]


    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Retention Priority",

        retention_priority

    ]


    prediction_report.loc[
        len(prediction_report)
    ] = [

        "Recommended Action",

        retention_recommendation

    ]


    # --------------------------------------------------
    # DISPLAY REPORT PREVIEW
    # --------------------------------------------------

    st.dataframe(
        prediction_report,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------
    # CONVERT REPORT TO CSV
    # --------------------------------------------------

    report_csv = (
        prediction_report.to_csv(
            index=False
        )
    )


    # --------------------------------------------------
    # CSV DOWNLOAD BUTTON
    # --------------------------------------------------

    st.download_button(
        label="📥 Download Customer Report",
        data=report_csv,
        file_name=(
            "customer_churn_prediction_report.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


    # ==================================================
    # GENERATE PDF REPORT
    # ==================================================

    pdf_report = create_pdf_report(

        credit_score=credit_score,

        geography=geography,

        gender=gender,

        age=age,

        tenure=tenure,

        balance=balance,

        num_of_products=num_of_products,

        has_cr_card_value=has_cr_card_value,

        is_active_member_value=is_active_member_value,

        estimated_salary=estimated_salary,

        churn_probability=churn_probability,

        churn_status=churn_status,

        risk_category=risk_category,

        estimated_ltv=estimated_ltv,

        retention_cost=retention_cost,

        expected_retention_value=expected_retention_value,

        expected_net_retention_value=(
            expected_net_retention_value
        ),

        retention_priority=retention_priority,

        retention_recommendation=(
            retention_recommendation
        ),

        final_model_name=final_model_name
    )


    # ==================================================
    # PDF DOWNLOAD BUTTON
    # ==================================================

    st.download_button(

        label="📄 Download PDF Report",

        data=pdf_report,

        file_name=(
            "customer_churn_prediction_report.pdf"
        ),

        mime="application/pdf",

        use_container_width=True
    )