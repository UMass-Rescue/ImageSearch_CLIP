from db import *
from model import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Main script for searching FAISS index
dataset_name = "coco"  # Example dataset name
query = "man wearing red shirt"

# Generate the query embedding
clip_model = CLIPModel()

clip_model.search()
