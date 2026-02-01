
from huggingface_hub.utils import RepositoryNotFoundError,HfHubHTTPError
from huggingface_hub import HfApi,create_repo

repo_id = "Vvaannii33/Tourism-Package-Creation"
repo_type = "dataset"

# Initialize API client

api=HfApi(token=os.getenv("TOURISM_TOKEN"))

# Step 1: Check if the space exists

try:
  api.repo_info(repo_id=repo_id,repo_type=repo_type)
  print(f"Space '{repo_id} already exists. Using it")
except:
  RepositoryNotFoundError
  print(f"Space '{repo_id}' does not exist. Creating it")
  create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
  print(f"Space '{repo_id}' is created")


#Upload the folder to Hugging face 

api.upload_folder(
    
           folder_path="tourism/data",
           repo_id=repo_id,
           repo_type=repo_type,
)
