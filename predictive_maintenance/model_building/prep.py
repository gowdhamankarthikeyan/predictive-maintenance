import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

# Read credentials from environment variables
HF_TOKEN        = os.getenv("HF_TOKEN")
HF_USERNAME     = os.getenv("HF_USERNAME",     "gowdhamankarthikeyan")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "engine-predictive-maintenance")

# Initialize HuggingFace API client
api = HfApi(token=HF_TOKEN)

# Step 1: Load raw dataset directly from HuggingFace dataset repo
dataset_path = f"hf://datasets/{HF_USERNAME}/{HF_DATASET_NAME}/data/engine_data_raw.csv"
df = pd.read_csv(dataset_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Feature Engineering — same three features as notebook
# Feature 1: rpm_x_fuel_pressure encodes the "struggling engine" signature
# Healthy engine: high RPM × moderate fuel = LARGE value
# Failing engine: low RPM × high fuel     = SMALL value
df['rpm_x_fuel_pressure'] = df['Engine_RPM'] * df['Fuel_Pressure']

# Feature 2: oil_health_index — viscosity degradation proxy (Pressure / Temperature)
# As oil degrades: pressure drops AND temperature rises simultaneously
df['oil_health_index'] = df['Lub_Oil_Pressure'] / df['Lub_Oil_Temp']

# Feature 3: rpm_bins — RPM operating regime (0=Idle, 1=Normal, 2=High-Load)
# Faulty engines predominantly in Idle regime (bin=0)
df['rpm_bins'] = pd.cut(
    df['Engine_RPM'],
    bins=[0, 500, 1000, float('inf')],
    labels=[0, 1, 2]
).astype(int)

print("Engineered features added: rpm_x_fuel_pressure, oil_health_index, rpm_bins")

# Step 3: Define features and target
feature_cols = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
    'Coolant_Pressure', 'Lub_Oil_Temp', 'Coolant_Temp',
    'rpm_x_fuel_pressure', 'oil_health_index', 'rpm_bins'
]
X = df[feature_cols]
y = df['Engine_Condition']

# Step 4: Stratified 75/10/15 split — preserves 63:37 class ratio across all splits
# First separate 15% test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
# From remaining 85%, split ~11.76% as validation → gives exactly 10% overall
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.10/0.85, stratify=y_temp, random_state=42
)

# Step 5: Combine features and target, save splits locally
train_df = X_train.copy(); train_df['Engine_Condition'] = y_train.values
val_df   = X_val.copy();   val_df['Engine_Condition']   = y_val.values
test_df  = X_test.copy();  test_df['Engine_Condition']  = y_test.values

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv",     index=False)
test_df.to_csv("test.csv",   index=False)

print(f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Step 6: Upload all three splits back to HuggingFace dataset repo
for fname in ["train.csv", "val.csv", "test.csv"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=f"data/{fname}",
        repo_id=f"{HF_USERNAME}/{HF_DATASET_NAME}",
        repo_type="dataset",
    )
    print(f"Uploaded {fname} to HuggingFace.")
