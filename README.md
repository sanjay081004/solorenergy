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
<img width="1912" height="867" alt="image" src="https://github.com/user-attachments/assets/465eb16b-e9b6-44e4-a56a-87bbbf18c12f" />
<img width="1907" height="902" alt="image" src="https://github.com/user-attachments/assets/dcde24bd-b4b1-4146-b536-372efabfbc36" />
<img width="1917" height="811" alt="image" src="https://github.com/user-attachments/assets/71eb10c9-c8ec-4d85-aa8c-7875859e4ad1" />
