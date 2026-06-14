import pandas as pd
import pickle
import mlflow
import os
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "gowdhamankarthikeyan")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "engine-predictive-maintenance")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "engine-maintenance-predictor")

api = HfApi(token=HF_TOKEN)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Predictive_Maintenance_Engine")

# Load train and test splits from HuggingFace
base = f"hf://datasets/{HF_USERNAME}/{HF_DATASET_NAME}/data"
train_df = pd.read_csv(f"{base}/train.csv")
test_df  = pd.read_csv(f"{base}/test.csv")

feature_cols = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
    'Coolant_Pressure', 'Lub_Oil_Temp', 'Coolant_Temp',
    'rpm_x_fuel_pressure', 'oil_health_index', 'rpm_bins'
]
X_train = train_df[feature_cols]
y_train = train_df['Engine_Condition']
X_test  = test_df[feature_cols]
y_test  = test_df['Engine_Condition']

# Best parameters from notebook tuning
best_params = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'reg_lambda': 2.0,
    'reg_alpha': 0.1,
    'colsample_bytree': 0.6,
    'random_state': 42,
    'eval_metric': 'logloss'
}

with mlflow.start_run(run_name="XGBoost_Tuned_Deploy"):
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy  = accuracy_score(y_test, y_pred)

    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "test_recall": recall,
        "test_f1": f1,
        "test_precision": precision,
        "test_accuracy": accuracy
    })

    print(f"Test Recall: {recall:.4f} | F1: {f1:.4f} | Precision: {precision:.4f}")

    model_path = "best_model.pkl"
    with open(model_path, 'wb') as f:
      pickle.dump(model, f)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved: {model_path}")

# Upload model to HuggingFace model repo
repo_id = f"{HF_USERNAME}/{HF_MODEL_NAME}"
try:
    api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)
    print(f"Model repo '{repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type="model",
)
print(f"Model uploaded to HuggingFace: {repo_id}")
