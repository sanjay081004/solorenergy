# Solar Power Generation Prediction

Streamlit dashboard that predicts plant-level AC power (kW) from weather conditions.

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
# put the two Plant_1 CSVs in ./data
python train_model.py
streamlit run streamlit_app.py
```
Optional: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` to use the Snowflake-registered model.
