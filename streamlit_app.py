import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Power | ML Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "HOUR", "DAY_OF_WEEK"]
# Snowflake DAYOFWEEK(): 0 = Sunday ... 6 = Saturday (what the model was trained on)
DAY_LABELS = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
              4: "Thursday", 5: "Friday", 6: "Saturday"}
REGISTRY_METRICS = {"MAE": 246.22, "RMSE": 501.44, "R2": 0.9960}  # registered V1 metrics

# ---------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(180deg, #f5f9fc 0%, #eef4f7 100%); color: #17324d; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
    .hero {
        padding: 2rem 2.2rem; border-radius: 22px; color: white;
        background: linear-gradient(115deg, #103b4b 0%, #087e78 58%, #55a66c 100%);
        box-shadow: 0 12px 32px rgba(16, 59, 75, .16); margin-bottom: 1.4rem;
    }
    .hero h1 { color: white !important; font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.65rem); font-weight: 800; margin: 0 0 .45rem 0; }
    .hero p { color: #e1f4ef !important; margin: .2rem 0; font-size: 1rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem;
        font-weight: 700; color: #b9e7d8; margin-bottom: .65rem; }
    .section-note { color: #60758a; font-size: .92rem; margin-top: -.45rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e2ebf0;
        padding: 1rem 1.1rem; border-radius: 16px; box-shadow: 0 5px 18px rgba(28, 61, 83, .045); }
    div[data-testid="stForm"] { background: white; border: 1px solid #e2ebf0;
        border-radius: 18px; padding: 1.2rem 1.35rem; }
    .result-panel { background: white; border: 1px solid #dce9e9; border-left: 5px solid #0d8b78;
        border-radius: 16px; padding: 1.25rem 1.5rem; margin: .8rem 0 1rem 0; }
    .result-label { color: #60758a; font-size: .85rem; font-weight: 600; }
    .result-value { color: #087e78; font-family: 'Manrope', sans-serif;
        font-size: 2.35rem; font-weight: 800; line-height: 1.25; }
    .footer { text-align: center; color: #718397; font-size: .82rem; padding: 1.4rem 0 .5rem 0; }
    [data-testid="stSidebar"] { background: #edf4f5; }
    </style>
    """,
    unsafe_allow_html=True,
)



def train_from_csv():
    """Same chronological 80/20 Random Forest as the notebook, from solar_training_data.csv."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if not Path("solar_training_data.csv").exists():
        raise FileNotFoundError(
            "Neither a usable model.joblib nor solar_training_data.csv was found. "
            "Run `python train_model.py` and commit the files it creates."
        )
    df = pd.read_csv("solar_training_data.csv", parse_dates=["DATE_TIME"]).sort_values("DATE_TIME")
    split = int(len(df) * 0.8)
    X, y = df[FEATURES], df["AC_POWER"]
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X.iloc[:split], y.iloc[:split])
    pred = rf.predict(X.iloc[split:])
    yt = y.iloc[split:]
    metrics = {"MAE": float(mean_absolute_error(yt, pred)),
               "RMSE": float(np.sqrt(mean_squared_error(yt, pred))),
               "R2": float(r2_score(yt, pred))}
    test_df = pd.DataFrame({"DATE_TIME": df["DATE_TIME"].iloc[split:].values,
                            "ACTUAL": yt.values, "PREDICTED": pred})
    return rf, test_df, metrics


# ---------------------------------------------------------------------
# BACKEND: Snowflake registry model (if secrets exist) or local model.joblib
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_backend():
    try:
        use_snowflake = "snowflake" in st.secrets
    except Exception:
        use_snowflake = False

    if use_snowflake:
        from snowflake.snowpark import Session
        from snowflake.ml.registry import Registry

        cfg = dict(st.secrets["snowflake"])
        session = Session.builder.configs(cfg).create()
        mv = Registry(
            session=session,
            database_name=cfg.get("database", "SOLAR_ML_DB"),
            schema_name=cfg.get("schema", "PUBLIC"),
        ).get_model("SOLAR_POWER_MODEL").version("V1")

        def predict(df: pd.DataFrame) -> pd.Series:
            out = mv.run(df[FEATURES], function_name="PREDICT")
            col = "output_feature_0" if "output_feature_0" in out.columns else out.columns[0]
            return out[col].astype(float).reset_index(drop=True)

        def history() -> pd.DataFrame:
            h = session.sql(
                """
                SELECT DATE_TIME, AC_POWER, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE,
                       IRRADIATION, HOUR, DAY_OF_WEEK
                FROM SOLAR_ML_DB.PUBLIC.SOLAR_TRAINING_DATA
                WHERE AC_POWER IS NOT NULL AND AMBIENT_TEMPERATURE IS NOT NULL
                  AND MODULE_TEMPERATURE IS NOT NULL AND IRRADIATION IS NOT NULL
                ORDER BY DATE_TIME DESC LIMIT 100
                """
            ).to_pandas()
            h["PREDICTED_AC_POWER"] = predict(h).values
            return h.rename(columns={"AC_POWER": "ACTUAL_AC_POWER"})

        return {"predict": predict, "history": history, "name": "Random Forest",
                "source": "Snowflake Model Registry · SOLAR_POWER_MODEL V1",
                "metrics": REGISTRY_METRICS}

    # ---- local / Streamlit Cloud mode (no Snowflake session needed) ----
    model, test_df, name, metrics = None, None, "Random Forest", REGISTRY_METRICS
    if Path("model.joblib").exists():
        try:
            model = joblib.load("model.joblib")  # may fail if scikit-learn versions differ
            meta = json.loads(Path("metrics.json").read_text()) if Path("metrics.json").exists() else None
            name = meta["best"] if meta else type(model).__name__
            metrics = meta["all"][name] if meta else REGISTRY_METRICS
        except Exception:
            model = None
    if model is None:  # fallback: train quickly from the committed training table
        model, test_df, metrics = train_from_csv()
        source = "Trained at startup from solar_training_data.csv"
    else:
        source = "Saved model.joblib"

    def predict(df: pd.DataFrame) -> pd.Series:
        return pd.Series(model.predict(df[FEATURES]), dtype=float)

    def history() -> pd.DataFrame:
        if test_df is not None:
            h = test_df.copy()
        elif Path("test_predictions.csv").exists():
            h = pd.read_csv("test_predictions.csv", parse_dates=["DATE_TIME"])
        else:
            raise FileNotFoundError("test_predictions.csv not found. Run `python train_model.py`.")
        h = h.tail(100)
        return h.rename(columns={"ACTUAL": "ACTUAL_AC_POWER", "PREDICTED": "PREDICTED_AC_POWER"})

    return {"predict": predict, "history": history, "name": name,
            "source": source, "metrics": metrics}


try:
    backend = load_backend()
except Exception as exc:
    st.error("Could not load the model. Run `python train_model.py` first, or check your "
             "Snowflake credentials in .streamlit/secrets.toml.")
    st.exception(exc)
    st.stop()

metrics = backend["metrics"]

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning • Random Forest Regression</div>
        <h1>☀️ Solar Power Generation Prediction</h1>
        <p>Estimate plant-level solar AC power from weather and time conditions.</p>
        <p>Enter the operating conditions, run the model, and explore its performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_col, model_col = st.columns([1, 2])
with status_col:
    st.success(f"Trained {backend['name']} model loaded")
with model_col:
    st.caption(f"Source: {backend['source']}")

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ☀️ Project Overview")
    st.write("Predicts total AC power generation of a solar plant from ambient temperature, "
             "module temperature, irradiation, hour and day of week.")
    st.markdown("---")
    st.markdown("**Model details**")
    st.write(f"Algorithm: {backend['name']}")
    st.write("Dataset: Kaggle Solar Power Generation (Plant 1)")
    st.write("Target: AC power summed over all inverters (kW)")
    st.markdown("---")
    st.markdown("**Input guidance**")
    st.caption("Temperatures in °C. Irradiation in kW/m², same scale as the training data.")
    st.caption("Day numbering follows Snowflake DAYOFWEEK(): Sunday = 0 ... Saturday = 6.")

# ---------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------
st.subheader("Model Performance")
st.markdown('<p class="section-note">Evaluation metrics on the held-out, most recent 20% of the timeline.</p>',
            unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("R² Score", f"{metrics['R2']:.4f}", help="Coefficient of determination")
m2.metric("RMSE", f"{metrics['RMSE']:,.2f} kW", help="Root mean squared error")
m3.metric("MAE", f"{metrics['MAE']:,.2f} kW", help="Mean absolute error")

st.divider()

# ---------------------------------------------------------------------
# PREDICTION FORM
# ---------------------------------------------------------------------
st.subheader("Predict Solar Power")
st.markdown('<p class="section-note">Provide the weather conditions and time details below.</p>',
            unsafe_allow_html=True)

with st.form("solar_prediction_form"):
    left, right = st.columns(2)
    with left:
        ambient = st.number_input("Ambient Temperature (°C)", 15.0, 45.0, 28.0, 0.5,
                                  help="Temperature of the surrounding air.")
        module = st.number_input("Module Temperature (°C)", 15.0, 75.0, 40.0, 0.5,
                                 help="Temperature of the solar panel/module.")
        irradiation = st.number_input("Irradiation (kW/m²)", 0.0, 1.3, 0.60, 0.01, format="%.2f",
                                      help="Training data ranges roughly from 0 to 1.2.")
    with right:
        hour = st.slider("Hour of Day", 0, 23, 12,
                         help="24-hour clock: 0 = midnight, 12 = noon, 23 = 11 PM.")
        day = st.selectbox("Day of Week", options=list(DAY_LABELS.keys()),
                           format_func=lambda v: DAY_LABELS[v], index=1)
        st.info(f"Selected time: {DAY_LABELS[day]}, {hour:02d}:00", icon="🕒")

    submitted = st.form_submit_button("⚡ Predict Solar Power", type="primary", width="stretch")

# ---------------------------------------------------------------------
# RESULT
# ---------------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame({
        "AMBIENT_TEMPERATURE": [float(ambient)],
        "MODULE_TEMPERATURE": [float(module)],
        "IRRADIATION": [float(irradiation)],
        "HOUR": [int(hour)],
        "DAY_OF_WEEK": [int(day)],
    })

    try:
        with st.spinner("Running prediction..."):
            predicted_power = max(0.0, float(backend["predict"](input_data).iloc[0]))

        st.divider()
        st.subheader("Prediction Result")
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">PREDICTED AC POWER (PLANT TOTAL)</div>
                <div class="result-value">{predicted_power:,.2f} kW</div>
                <div class="result-label">Model: {backend['name']} • {backend['source']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge_max = max(35000.0, predicted_power * 1.1)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_power,
            number={"suffix": " kW", "valueformat": ",.2f"},
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
        ))
        gauge.update_layout(height=330, margin=dict(l=25, r=25, t=55, b=15),
                            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#17324d"})
        st.plotly_chart(gauge, width="stretch")

        with st.expander("View Input Details", expanded=True):
            summary = input_data.copy()
            summary["DAY_NAME"] = DAY_LABELS[int(day)]
            st.dataframe(summary, width="stretch", hide_index=True)

        st.download_button(
            "Download prediction as CSV",
            data=summary.assign(PREDICTED_AC_POWER_KW=predicted_power).to_csv(index=False),
            file_name="solar_power_prediction.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error("Prediction failed. Check the model and input feature names.")
        st.exception(exc)

# ---------------------------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------------------------
st.divider()
st.subheader("Historical Actual vs Predicted")
st.markdown('<p class="section-note">Actual AC power compared with model predictions for 100 '
            'held-out records.</p>', unsafe_allow_html=True)

if st.button("Load historical comparison"):
    try:
        with st.spinner("Generating comparison..."):
            hist = backend["history"]()
            hist["DATE_TIME"] = pd.to_datetime(hist["DATE_TIME"])
            hist = hist.sort_values("DATE_TIME")

        chart_data = hist.melt(
            id_vars=["DATE_TIME"],
            value_vars=["ACTUAL_AC_POWER", "PREDICTED_AC_POWER"],
            var_name="Series", value_name="AC Power (kW)",
        )
        chart_data["Series"] = chart_data["Series"].map(
            {"ACTUAL_AC_POWER": "Actual AC Power", "PREDICTED_AC_POWER": "Predicted AC Power"}
        )
        fig = px.line(
            chart_data, x="DATE_TIME", y="AC Power (kW)", color="Series",
            title="Actual vs Predicted AC Power (100 records)",
            color_discrete_map={"Actual AC Power": "#123b56", "Predicted AC Power": "#0d9b83"},
        )
        fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                          legend_title_text="", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")

        actual = hist["ACTUAL_AC_POWER"].astype(float)
        pred = hist["PREDICTED_AC_POWER"].astype(float)
        c1, c2, c3 = st.columns(3)
        c1.metric("Records compared", f"{len(hist):,}")
        c2.metric("Sample MAE", f"{(actual - pred).abs().mean():,.2f} kW")
        c3.metric("Sample RMSE", f"{(((actual - pred) ** 2).mean()) ** 0.5:,.2f} kW")
        st.caption("Metrics for this sample only; they do not replace the overall evaluation above.")
    except Exception as exc:
        st.error("Historical comparison could not be generated. Predictions still work independently.")
        st.exception(exc)

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Solar Power Generation Prediction • Academic Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)
