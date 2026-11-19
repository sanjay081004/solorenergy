"""
Rebuilds the same training table as your SOLAR_TRAINING_DATA view and trains the
same three models you compared in the notebook (chronological 80/20 split).
Saves the best one to model.joblib so the Streamlit app can run offline.

Put these two Kaggle files in ./data :
    Plant_1_Generation_Data.csv
    Plant_1_Weather_Sensor_Data.csv
Run:  python train_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

DATA = Path("data")
FEATURES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "HOUR", "DAY_OF_WEEK"]

gen = pd.read_csv(DATA / "Plant_1_Generation_Data.csv")
wea = pd.read_csv(DATA / "Plant_1_Weather_Sensor_Data.csv")

# The two Kaggle files use different date formats
gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], format="%d-%m-%Y %H:%M")
wea["DATE_TIME"] = pd.to_datetime(wea["DATE_TIME"])

# Same logic as the SQL view: SUM AC power over all inverters, AVG weather, join on timestamp
g = gen.groupby("DATE_TIME", as_index=False)["AC_POWER"].sum()
w = wea.groupby("DATE_TIME", as_index=False)[
    ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
].mean()
df = g.merge(w, on="DATE_TIME", how="inner").sort_values("DATE_TIME").reset_index(drop=True)

df["HOUR"] = df["DATE_TIME"].dt.hour
# Snowflake DAYOFWEEK: 0 = Sunday ... 6 = Saturday (pandas is Monday = 0)
df["DAY_OF_WEEK"] = (df["DATE_TIME"].dt.dayofweek + 1) % 7
df[["DATE_TIME"] + FEATURES + ["AC_POWER"]].to_csv("solar_training_data.csv", index=False)
print("Training table shape:", df.shape, "(your Snowflake view had 3157 rows)")

split = int(len(df) * 0.8)
X, y = df[FEATURES], df["AC_POWER"]
X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

candidates = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
}

results, fitted = {}, {}
for name, m in candidates.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    results[name] = {
        "MAE": float(mean_absolute_error(y_te, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, p))),
        "R2": float(r2_score(y_te, p)),
    }
    fitted[name] = m
    print(f"{name:18s} MAE={results[name]['MAE']:.2f}  RMSE={results[name]['RMSE']:.2f}  R2={results[name]['R2']:.4f}")

best = min(results, key=lambda k: results[k]["RMSE"])
print("Best model:", best)

joblib.dump(fitted[best], "model.joblib")
Path("metrics.json").write_text(json.dumps({"best": best, "all": results}, indent=2))

pd.DataFrame({
    "DATE_TIME": df["DATE_TIME"].iloc[split:].values,
    "ACTUAL": y_te.values,
    "PREDICTED": fitted[best].predict(X_te),
}).to_csv("test_predictions.csv", index=False)
print("Saved model.joblib, metrics.json, test_predictions.csv, solar_training_data.csv")
