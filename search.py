from model import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Main script for searching FAISS index
dataset_name = "coco"  # Example dataset name
query = "cat"

# Generate the query embedding
clip_model = CLIPModel()
query_embedding = clip_model.generate_text_embedding(query)