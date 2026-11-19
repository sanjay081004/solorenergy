import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry


# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Power | ML Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #f5f9fc 0%, #eef4f7 100%);
        color: #17324d;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        color: white;
        background: linear-gradient(115deg, #103b4b 0%, #087e78 58%, #55a66c 100%);
        box-shadow: 0 12px 32px rgba(16, 59, 75, .16);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        color: white !important;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        font-weight: 800;
        margin: 0 0 .45rem 0;
    }
    .hero p {
        color: #e1f4ef !important;
        margin: .2rem 0;
        font-size: 1rem;
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .72rem;
        font-weight: 700;
        color: #b9e7d8;
        margin-bottom: .65rem;
    }
    .section-note {
        color: #60758a;
        font-size: .92rem;
        margin-top: -.45rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2ebf0;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(28, 61, 83, .045);
    }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e2ebf0;
        border-radius: 18px;
        padding: 1.2rem 1.35rem;
    }
    .result-panel {
        background: white;
        border: 1px solid #dce9e9;
        border-left: 5px solid #0d8b78;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: .8rem 0 1rem 0;
    }
    .result-label {
        color: #60758a;
        font-size: .85rem;
        font-weight: 600;
    }
    .result-value {
        color: #087e78;
        font-family: 'Manrope', sans-serif;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.25;
    }
    .footer {
        text-align: center;
        color: #718397;
        font-size: .82rem;
        padding: 1.4rem 0 .5rem 0;
    }
    [data-testid="stSidebar"] {
        background: #edf4f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# SNOWFLAKE MODEL CONNECTION
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Snowflake ML...")
def load_model():
    """Load the existing registered model from the active Snowflake session."""
    session = get_active_session()
    session.sql("USE DATABASE SOLAR_ML_DB").collect()
    session.sql("USE SCHEMA PUBLIC").collect()

    registry = Registry(
        session=session,
        database_name="SOLAR_ML_DB",
        schema_name="PUBLIC",
    )
    model_version = registry.get_model("SOLAR_POWER_MODEL").version("V1")
    return session, model_version


try:
    session, model = load_model()
    connection_ok = True
except Exception as exc:
    connection_ok = False
    st.error(
        "Could not connect to SOLAR_POWER_MODEL version V1. "
        "Confirm that this app is running in the Snowflake warehouse runtime "
        "and that its owner role can use the model and database."
    )
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Snowflake ML • Random Forest Regression</div>
        <h1>☀️ Solar Power Generation Prediction</h1>
        <p>Estimate solar AC power from real-world weather and time conditions.</p>
        <p>Enter the operating conditions, run the registered model, and explore
        historical model performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_col, model_col = st.columns([1, 2])
with status_col:
    st.success("Snowflake connection active")
with model_col:
    st.caption("Model: SOLAR_ML_DB.PUBLIC.SOLAR_POWER_MODEL  |  Version: V1")


# ---------------------------------------------------------------------
# SIDEBAR / PROJECT DETAILS
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ☀️ Project Overview")
    st.write(
        "A machine-learning application that predicts AC power generation "
        "from ambient temperature, module temperature, irradiation, hour, "
        "and day of week."
    )
    st.markdown("---")
    st.markdown("**Model details**")
    st.write("Algorithm: Random Forest Regressor")
    st.write("Platform: Snowflake Model Registry")
    st.write("Model version: V1")
    st.markdown("---")
    st.markdown("**Input guidance**")
    st.caption("Temperatures are entered in °C. Irradiation uses the same scale "
               "as the training dataset.")
    st.caption("Day numbering follows Snowflake DAYOFWEEK(): Sunday = 0, "
               "Monday = 1, ..., Saturday = 6.")


# ---------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------
st.subheader("Model Performance")
st.markdown(
    '<p class="section-note">Evaluation metrics recorded for the registered model.</p>',
    unsafe_allow_html=True,
)
m1, m2, m3 = st.columns(3)
m1.metric("R² Score", "0.9960", help="Coefficient of determination")
m2.metric("RMSE", "501.44 W", help="Root mean squared error")
m3.metric("MAE", "246.22 W", help="Mean absolute error")

st.divider()


# ---------------------------------------------------------------------
# PREDICTION INPUT FORM
# ---------------------------------------------------------------------
st.subheader("Predict Solar Power")
st.markdown(
    '<p class="section-note">Provide the weather conditions and time details below.</p>',
    unsafe_allow_html=True,
)

day_labels = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

with st.form("solar_prediction_form"):
    left, right = st.columns(2)

    with left:
        ambient = st.number_input(
            "Ambient Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
            help="Temperature of the surrounding air.",
        )
        module = st.number_input(
            "Module Temperature (°C)",
            min_value=-20.0,
            max_value=100.0,
            value=30.0,
            step=0.5,
            help="Temperature of the solar panel/module.",
        )
        irradiation = st.number_input(
            "Irradiation",
            min_value=0.0,
            max_value=2.0,
            value=0.50,
            step=0.01,
            format="%.2f",
            help="Use the same irradiation units and scale as the training data.",
        )

    with right:
        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12,
            help="24-hour clock: 0 = midnight, 12 = noon, 23 = 11 PM.",
        )
        day = st.selectbox(
            "Day of Week",
            options=list(day_labels.keys()),
            format_func=lambda value: day_labels[value],
            index=1,
        )
        st.info(
            f"Selected time: {day_labels[day]}, {hour:02d}:00",
            icon="🕒",
        )

    submitted = st.form_submit_button(
        "⚡ Predict Solar Power",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# PREDICTION RESULT
# ---------------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame(
        {
            "AMBIENT_TEMPERATURE": [float(ambient)],
            "MODULE_TEMPERATURE": [float(module)],
            "IRRADIATION": [float(irradiation)],
            "HOUR": [int(hour)],
            "DAY_OF_WEEK": [int(day)],
        }
    )

    try:
        with st.spinner("Running prediction with SOLAR_POWER_MODEL V1..."):
            prediction_df = model.run(input_data, function_name="PREDICT")

        if "output_feature_0" not in prediction_df.columns:
            raise ValueError(
                "The model response does not contain output_feature_0. "
                f"Returned columns: {list(prediction_df.columns)}"
            )

        predicted_power = float(prediction_df.iloc[0]["output_feature_0"])
        predicted_power = max(0.0, predicted_power)

        st.divider()
        st.subheader("Prediction Result")
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">PREDICTED AC POWER</div>
                <div class="result-value">{predicted_power:,.2f} W</div>
                <div class="result-label">Generated by registered model SOLAR_POWER_MODEL • V1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge_max = max(1000.0, predicted_power * 1.2)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_power,
                number={"suffix": " W", "valueformat": ",.2f"},
                title={"text": "Predicted AC Power"},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "#0d8b78"},
                    "bgcolor": "#edf4f5",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, gauge_max * 0.5], "color": "#e8f2ef"},
                        {"range": [gauge_max * 0.5, gauge_max], "color": "#d6ebe4"},
                    ],
                },
            )
        )
        gauge.update_layout(
            height=330,
            margin=dict(l=25, r=25, t=55, b=15),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#17324d"},
        )
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("**Submitted weather conditions**")
        summary = input_data.copy()
        summary["DAY_NAME"] = day_labels[int(day)]
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.download_button(
            "Download prediction as CSV",
            data=summary.assign(PREDICTED_AC_POWER_W=predicted_power).to_csv(index=False),
            file_name="solar_power_prediction.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error("Prediction failed. Check model permissions and input feature names.")
        st.exception(exc)


# ---------------------------------------------------------------------
# ACTUAL VS PREDICTED HISTORICAL VISUALIZATION
# ---------------------------------------------------------------------
st.divider()
st.subheader("Historical Actual vs Predicted")
st.markdown(
    '<p class="section-note">Compares the registered model predictions with actual AC power '
    'for a small sample from SOLAR_TRAINING_DATA.</p>',
    unsafe_allow_html=True,
)

if st.button("Load historical comparison", use_container_width=False):
    try:
        with st.spinner("Loading recent records and generating comparison..."):
            history = session.sql(
                """
                SELECT
                    DATE_TIME,
                    AC_POWER,
                    AMBIENT_TEMPERATURE,
                    MODULE_TEMPERATURE,
                    IRRADIATION,
                    HOUR,
                    DAY_OF_WEEK
                FROM SOLAR_ML_DB.PUBLIC.SOLAR_TRAINING_DATA
                WHERE AC_POWER IS NOT NULL
                  AND AMBIENT_TEMPERATURE IS NOT NULL
                  AND MODULE_TEMPERATURE IS NOT NULL
                  AND IRRADIATION IS NOT NULL
                ORDER BY DATE_TIME DESC
                LIMIT 100
                """
            ).to_pandas()

            if history.empty:
                st.warning("No historical records were returned from SOLAR_TRAINING_DATA.")
            else:
                feature_columns = [
                    "AMBIENT_TEMPERATURE",
                    "MODULE_TEMPERATURE",
                    "IRRADIATION",
                    "HOUR",
                    "DAY_OF_WEEK",
                ]
                historical_predictions = model.run(
                    history[feature_columns],
                    function_name="PREDICT",
                )

                history["PREDICTED_AC_POWER"] = historical_predictions[
                    "output_feature_0"
                ].astype(float).values
                history["DATE_TIME"] = pd.to_datetime(history["DATE_TIME"])
                history = history.sort_values("DATE_TIME")

                chart_data = history.melt(
                    id_vars=["DATE_TIME"],
                    value_vars=["AC_POWER", "PREDICTED_AC_POWER"],
                    var_name="Series",
                    value_name="AC Power (W)",
                )
                chart_data["Series"] = chart_data["Series"].map(
                    {
                        "AC_POWER": "Actual AC Power",
                        "PREDICTED_AC_POWER": "Predicted AC Power",
                    }
                )

                fig = px.line(
                    chart_data,
                    x="DATE_TIME",
                    y="AC Power (W)",
                    color="Series",
                    markers=False,
                    title="Actual vs Predicted AC Power (latest 100 records)",
                    color_discrete_map={
                        "Actual AC Power": "#123b56",
                        "Predicted AC Power": "#0d9b83",
                    },
                )
                fig.update_layout(
                    height=420,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="white",
                    legend_title_text="",
                    margin=dict(l=10, r=10, t=55, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                actual = history["AC_POWER"].astype(float)
                predicted = history["PREDICTED_AC_POWER"].astype(float)
                compare1, compare2, compare3 = st.columns(3)
                compare1.metric("Records compared", f"{len(history):,}")
                compare2.metric("Sample MAE", f"{(actual - predicted).abs().mean():,.2f} W")
                compare3.metric(
                    "Sample RMSE",
                    f"{(((actual - predicted) ** 2).mean()) ** 0.5:,.2f} W",
                )
                st.caption(
                    "These are metrics for the displayed recent sample, not a replacement "
                    "for the registered model's evaluation metrics above."
                )

    except Exception as exc:
        st.error(
            "Historical comparison could not be generated. The prediction form can still "
            "work independently."
        )
        st.exception(exc)


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Solar Power Generation Prediction • Academic Machine Learning Project<br>
        Powered by Snowflake ML Model Registry
    </div>
    """,
    unsafe_allow_html=True,
)
