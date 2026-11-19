-- Run SHOW STREAMLITS first and replace the placeholder with the exact
-- Streamlit object name returned by Snowflake.
--
-- This switches the existing Streamlit app to warehouse runtime.
-- It does not recreate or modify your database tables, view, or ML model.

USE DATABASE SOLAR_ML_DB;
USE SCHEMA PUBLIC;

SHOW STREAMLITS IN SCHEMA SOLAR_ML_DB.PUBLIC;

-- Replace <STREAMLIT_APP_NAME> below with the exact app name from SHOW STREAMLITS.
ALTER STREAMLIT SOLAR_ML_DB.PUBLIC.<STREAMLIT_APP_NAME>
  SET
    RUNTIME_NAME = 'SYSTEM$WAREHOUSE_RUNTIME',
    QUERY_WAREHOUSE = COMPUTE_WH;

-- Verify the runtime and warehouse settings:
DESC STREAMLIT SOLAR_ML_DB.PUBLIC.<STREAMLIT_APP_NAME>;
