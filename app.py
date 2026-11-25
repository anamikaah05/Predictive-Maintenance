import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# =========================================================
# THEME
# =========================================================

def apply_theme(color):

    st.markdown(
        f"""
        <style>

        /* =================================================
           MAIN APPLICATION
           ================================================= */

        .stApp {{
            background-color: #F8F9FA;
            color: #212529;
        }}

        /* General text visibility */
        .stApp p,
        .stApp label {{
            color: #212529;
        }}

        /* Main headings */
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {{
            color: #111111 !important;
        }}


        /* =================================================
           SIDEBAR
           ================================================= */

        section[data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid #E9ECEF;
        }}

        section[data-testid="stSidebar"] label {{
            color: #212529 !important;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6 {{
            color: #212529 !important;
        }}


        /* =================================================
           SELECTBOX
           ================================================= */

        /* Selectbox main box */
        div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            color: #FFFFFF !important;
        }}

        /* Selected value */
        div[data-baseweb="select"] span {{
            color: #FFFFFF !important;
        }}

        /* Selectbox text */
        div[data-baseweb="select"] * {{
            color: #FFFFFF !important;
        }}

        /* Selectbox arrow */
        div[data-baseweb="select"] svg {{
            fill: #FFFFFF !important;
            color: #FFFFFF !important;
        }}

        /* Dropdown menu */
        ul[data-testid="stSelectboxVirtualDropdown"] {{
            background-color: #FFFFFF !important;
        }}

        ul[data-testid="stSelectboxVirtualDropdown"] li {{
            color: #212529 !important;
        }}


        /* =================================================
           NUMBER INPUT BOX
           ================================================= */

        /* Input field */
        div[data-testid="stNumberInput"] input {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            border: 1px solid #CED4DA !important;
            opacity: 1 !important;
        }}

        /* Input field when focused */
        div[data-testid="stNumberInput"] input:focus {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            border: 1px solid #6C757D !important;
        }}

        /* Number input buttons */
        div[data-testid="stNumberInput"] button {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-left: 1px solid #CED4DA !important;
            opacity: 1 !important;
        }}

        /* Arrow icons */
        div[data-testid="stNumberInput"] button svg {{
            fill: #000000 !important;
            color: #000000 !important;
            opacity: 1 !important;
        }}

        /* Arrow hover */
        div[data-testid="stNumberInput"] button:hover {{
            background-color: #F1F3F5 !important;
        }}


        /* =================================================
           MODEL TYPE CODE BOX
           ================================================= */

        /* Code box */
        code {{
            background-color: #212529 !important;
            color: #FFFFFF !important;
            padding: 3px 7px !important;
            border-radius: 5px !important;
        }}

        /* Model Type text */
        .stMarkdown code {{
            color: #FFFFFF !important;
        }}


        /* =================================================
           BUTTON
           ================================================= */

        .stButton>button {{
            background-color: {color};
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            border: none;
        }}

        .stButton>button:hover {{
            color: white !important;
        }}


        /* =================================================
           METRICS
           ================================================= */

        [data-testid="stMetricLabel"] {{
            color: #212529 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #111111 !important;
        }}

        [data-testid="stMetricDelta"] {{
            color: #212529 !important;
        }}


        /* =================================================
           MARKDOWN
           ================================================= */

        .stMarkdown {{
            color: #212529 !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TITLE
# =========================================================

st.title("⚙️Predictive Maintenance Dashboard")

st.write(
    "Select an ML algorithm and enter the machine parameters "
    "to generate a prediction."
)


# =========================================================
# MODEL SELECTION
# =========================================================

st.sidebar.header("🛠️ Model Selection")

algorithm = st.sidebar.selectbox(
    "Choose ML Algorithm:",
    [
        "K-Nearest Neighbors (KNN)",
        "Logistic Regression",
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ]
)


# =========================================================
# INPUT PARAMETERS
# =========================================================

st.sidebar.header("🔢 Machine Parameters")


machine_type_str = st.sidebar.selectbox(
    "Type",
    ["L", "M", "H"]
)


# =========================================================
# NUMBER INPUTS
# =========================================================

air_temp = st.sidebar.number_input(
    "Air temperature [K]",
    value=300.0,
    step=0.1
)


proc_temp = st.sidebar.number_input(
    "Process temperature [K]",
    value=310.0,
    step=0.1
)


rot_speed = st.sidebar.number_input(
    "Rotational speed [rpm]",
    value=1500,
    step=10
)


tool_wear = st.sidebar.number_input(
    "Tool wear [min]",
    value=50,
    step=1
)


# =========================================================
# TORQUE INPUT
#
# Only classification models need Torque as an input.
# Linear Regression predicts Torque.
# =========================================================

if algorithm != "Linear Regression":

    torque = st.sidebar.number_input(
        "Torque [Nm]",
        value=40.0,
        step=0.1
    )


# =========================================================
# TYPE ENCODING
# =========================================================

type_mapping = {
    "L": 0,
    "M": 1,
    "H": 2
}

machine_type_num = type_mapping[machine_type_str]


# =========================================================
# CLASSIFICATION MODEL INPUT
#
# KNN
# Logistic Regression
# Decision Tree
# Random Forest
#
# 6 FEATURES
# =========================================================

if algorithm != "Linear Regression":

    input_df = pd.DataFrame([{

        "Type": machine_type_num,

        "Air temperature [K]": air_temp,

        "Process temperature [K]": proc_temp,

        "Rotational speed [rpm]": rot_speed,

        "Torque [Nm]": torque,

        "Tool wear [min]": tool_wear

    }])


    # Keep EXACT same order used during training

    input_df = input_df[[
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]]


    input_data = input_df.values


# =========================================================
# LINEAR REGRESSION INPUT
#
# 5 FEATURES
#
# Target = Torque [Nm]
# =========================================================

else:

    linear_input_df = pd.DataFrame([{

        "Type": machine_type_num,

        "Air temperature [K]": air_temp,

        "Process temperature [K]": proc_temp,

        "Rotational speed [rpm]": rot_speed,

        "Tool wear [min]": tool_wear

    }])


    # EXACT 5 FEATURES USED DURING TRAINING

    linear_input_df = linear_input_df[[
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Tool wear [min]"
    ]]


    linear_input_data = linear_input_df.values


# =========================================================
# PROBABILITY FUNCTION
# =========================================================

def get_probability(model, data):

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(data)

        if probs.ndim == 2 and probs.shape[1] > 1:

            return probs[0, 1]

        return float(probs.ravel()[0])

    return None


# =========================================================
# LOAD MODEL AND PREDICT
# =========================================================

try:

    # =====================================================
    # KNN
    # =====================================================

    if algorithm == "K-Nearest Neighbors (KNN)":

        apply_theme("#E63946")

        model = joblib.load(
            "model(knn).pkl"
        )

        icon = "🕸️"

        tag = "KNN Classifier"

        pred = model.predict(
            input_data
        )[0]

        prob = get_probability(
            model,
            input_data
        )


    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    elif algorithm == "Logistic Regression":

        apply_theme("#1D3557")

        model = joblib.load(
            "model(log).pkl"
        )

        icon = "📈"

        tag = "Logistic Regression Classifier"

        pred = model.predict(
            input_data
        )[0]

        prob = get_probability(
            model,
            input_data
        )


    # =====================================================
    # LINEAR REGRESSION
    #
    # 5 FEATURES → TORQUE
    # =====================================================

    elif algorithm == "Linear Regression":

        apply_theme("#6F42C1")

        model = joblib.load(
            "model(lm).pkl"
        )

        icon = "📏"

        tag = "Linear Regression - Torque Prediction"

        pred_torque = model.predict(
            linear_input_data
        )[0]


    # =====================================================
    # DECISION TREE
    # =====================================================

    elif algorithm == "Decision Tree":

        apply_theme("#2A9D8F")

        model = joblib.load(
            "model(dt).pkl"
        )

        icon = "🌳"

        tag = "Decision Tree Classifier"

        pred = model.predict(
            input_data
        )[0]

        prob = get_probability(
            model,
            input_data
        )


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    elif algorithm == "Random Forest":

        apply_theme("#F4A261")

        model = joblib.load(
            "model(rf).pkl"
        )

        icon = "🌲🌲"

        tag = "Random Forest Classifier"

        pred = model.predict(
            input_data
        )[0]

        prob = get_probability(
            model,
            input_data
        )


# =========================================================
# ERROR HANDLING
# =========================================================

except Exception as e:

    st.error(
        f"Error loading or running model: {e}"
    )

    st.info(
        "Check that all .pkl files are inside the models folder."
    )

    st.stop()


# =========================================================
# DISPLAY RESULTS
# =========================================================

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    st.subheader(
        f"{icon} Active Model: {algorithm}"
    )

    st.write(
        f"**Model Type:** `{tag}`"
    )

    st.markdown("### Prediction Result")


    # =====================================================
    # LINEAR REGRESSION
    # =====================================================

    if algorithm == "Linear Regression":

        st.success(
            f"📏 Predicted Torque: "
            f"**{pred_torque:.2f} Nm**"
        )

        st.metric(
            label="Predicted Torque",
            value=f"{pred_torque:.2f} Nm"
        )


    # =====================================================
    # CLASSIFICATION MODELS
    # =====================================================

    else:

        if int(pred) == 1:

            st.error(
                "⚠️ ALERT: Machine Failure Predicted!"
            )

        else:

            st.success(
                "✅ STATUS: Normal Operation "
                "(No Failure)"
            )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    st.subheader(
        "📊 Prediction Visualization"
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )


    # =====================================================
    # KNN
    # =====================================================

    if algorithm == "K-Nearest Neighbors (KNN)":

        p_val = (
            prob
            if prob is not None
            else 0.5
        )

        ax.bar(
            ["Safe", "Failure"],
            [1 - p_val, p_val]
        )

        ax.set_title(
            "KNN Prediction Probability"
        )

        ax.set_ylabel(
            "Probability"
        )


    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    elif algorithm == "Logistic Regression":

        z = np.linspace(
            -6,
            6,
            100
        )

        sigmoid = 1 / (
            1 + np.exp(-z)
        )

        ax.plot(
            z,
            sigmoid,
            linewidth=2
        )

        ax.axvline(
            x=0,
            linestyle="--"
        )

        ax.set_title(
            "Logistic Regression Sigmoid Curve"
        )

        ax.set_xlabel(
            "Model Score"
        )

        ax.set_ylabel(
            "Probability"
        )


    # =====================================================
    # LINEAR REGRESSION
    # =====================================================

    elif algorithm == "Linear Regression":

        ax.bar(
            ["Predicted Torque"],
            [pred_torque]
        )

        ax.set_title(
            "Linear Regression Torque Prediction"
        )

        ax.set_ylabel(
            "Torque [Nm]"
        )


    # =====================================================
    # DECISION TREE
    # =====================================================

    elif algorithm == "Decision Tree":

        ax.barh(
            [
                "Tool Wear",
                "Rotational Speed"
            ],
            [
                tool_wear / 260,
                rot_speed / 2900
            ]
        )

        ax.set_title(
            "Decision Tree Input Features"
        )


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    elif algorithm == "Random Forest":

        p_val = (
            prob
            if prob is not None
            else 0.2
        )

        trees_conf = np.random.normal(
            p_val,
            0.05,
            50
        )

        ax.hist(
            trees_conf,
            bins=10,
            edgecolor="black"
        )

        ax.set_title(
            "Random Forest Tree Voting Spread"
        )

        ax.set_xlabel(
            "Failure Probability"
        )


    st.pyplot(fig)