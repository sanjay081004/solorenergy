# Solar Power Generation Prediction — Snowflake Streamlit

This project uses the existing registered Snowflake model:

- Database: `SOLAR_ML_DB`
- Schema: `PUBLIC`
- Model: `SOLAR_POWER_MODEL`
- Version: `V1`
- Prediction function: `PREDICT`

It does **not** recreate tables, views, or the model.

## Important deployment distinction

GitHub is used as the source-code repository. The app is intended to run **inside Snowflake Streamlit (warehouse runtime)** so it can use Snowflake's active session and the registered model. Pushing the code to GitHub alone does not host the live dashboard, and GitHub Pages cannot run this Python/Snowflake app.

Do not run `streamlit run streamlit_app.py` as the production deployment path: `get_active_session()` is provided by the Snowflake warehouse runtime.

## Files

- `streamlit_app.py` — complete dashboard.
- `environment.yml` — dependencies for Snowflake warehouse runtime.
- `deploy_runtime.sql` — SQL template to switch an existing Streamlit object to warehouse runtime.

## Deploy using your GitHub repository

1. Create a GitHub repository and upload `streamlit_app.py` and `environment.yml` to its root.
2. In Snowsight, open **Projects → Streamlit**.
3. Choose **Create from repository** (or connect your existing Streamlit app to the repository).
4. Select your Git repository, branch, and `streamlit_app.py`.
5. Choose database `SOLAR_ML_DB` and schema `PUBLIC`.
6. Choose an available virtual warehouse for the app's **Query warehouse** / **App warehouse**.
7. If the setup screen offers runtime selection, choose **Warehouse** / `SYSTEM$WAREHOUSE_RUNTIME`.
8. Create/open the app. The app uses `get_active_session()` and loads `SOLAR_POWER_MODEL` version `V1`.

The warehouse runtime runs app code on a virtual warehouse, unlike the container runtime which requires a compute pool. Snowflake's current documentation confirms both runtimes exist and that `SYSTEM$WAREHOUSE_RUNTIME` selects the warehouse runtime.

## If you already have a Streamlit object

1. Run `SHOW STREAMLITS IN SCHEMA SOLAR_ML_DB.PUBLIC;`
2. Identify the exact app name and confirm the role has permission to alter it.
3. Edit `deploy_runtime.sql`: replace `<STREAMLIT_APP_NAME>` with the exact object name returned by SHOW STREAMLITS.
4. Run the ALTER statement in Snowsight.

If your current Streamlit object belongs to a different database/schema, use that exact database/schema in the ALTER statement. Do not copy an object name from another account.

## Required Snowflake access

The role that owns/runs the app must be able to:
- use database `SOLAR_ML_DB` and schema `PUBLIC`;
- use the selected virtual warehouse;
- access the registered model `SOLAR_POWER_MODEL`, version `V1`;
- select from `SOLAR_TRAINING_DATA` for the optional historical comparison.

If the model loads but the historical chart fails, check SELECT permission on `SOLAR_TRAINING_DATA`. The prediction form is independent of the historical chart.

## Notes

- Day-of-week values intentionally follow Snowflake `DAYOFWEEK()`: Sunday=0, Monday=1, ..., Saturday=6. This fixes the Monday-first mapping in the earlier app.
- The displayed R², RMSE, and MAE values are the supplied registered-model metrics.
- The historical comparison is computed on the latest 100 rows and reports sample metrics; it is not a new model evaluation.
- Irradiation input range is a UI guardrail (0–2). For best predictions, enter values in the same scale as the Plant 1 training data.
- No password, private key, or Snowflake credential should be committed to GitHub.
