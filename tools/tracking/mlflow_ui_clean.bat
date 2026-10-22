@echo off
rem Serve the MLflow UI on the CLEAN result store (only the compared configurations), then open http://127.0.0.1:5001
rem Built by tools/tracking/build_clean_store.py --fresh.
set "MLFLOW_DISABLE_AGENT_HINT=1"
echo MLflow UI (clean store): http://127.0.0.1:5001
"%USERPROFILE%\anaconda3\envs\mlflow_tracking\Scripts\mlflow.exe" ui --backend-store-uri "sqlite:///D:/Aamir Gulzar/KSA_project2/mlflow_clean/mlflow.db" --host 127.0.0.1 --port 5001
