from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Read credentials from environment variables
HF_TOKEN      = os.getenv("HF_TOKEN")
HF_USERNAME   = os.getenv("HF_USERNAME",   "gowdhamankarthikeyan")
HF_SPACE_NAME = os.getenv("HF_SPACE_NAME", "engine-maintenance-app")

# Initialize HuggingFace API client
api     = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USERNAME}/{HF_SPACE_NAME}"

# Check if Space exists; create with Docker SDK if not
try:
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", private=False)
    print(f"Space '{repo_id}' created with Docker SDK.")

# Upload deployment folder (app.py, Dockerfile, requirements.txt) to Space root
api.upload_folder(
    folder_path="predictive_maintenance/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment files pushed to: https://huggingface.co/spaces/{repo_id}")
