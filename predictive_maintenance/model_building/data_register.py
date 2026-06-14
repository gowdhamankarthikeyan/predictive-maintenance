from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Read credentials and repo names from environment variables
# Set as GitHub Actions secrets; in Colab set via os.environ or userdata
HF_TOKEN        = os.getenv("HF_TOKEN")
HF_USERNAME     = os.getenv("HF_USERNAME",     "gowdhamankarthikeyan")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "engine-predictive-maintenance")

repo_id   = f"{HF_USERNAME}/{HF_DATASET_NAME}"
repo_type = "dataset"

# Initialize HuggingFace API client with token
api = HfApi(token=HF_TOKEN)

# Check if dataset repo exists; create it if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Upload the data folder (contains engine_data_raw.csv) to HuggingFace Hub
api.upload_folder(
    folder_path="predictive_maintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Raw dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
