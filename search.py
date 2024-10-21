from db import *
from model import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Main script for searching FAISS index
dataset_name = "coco"  # Example dataset name
query = "cat"

# Generate the query embedding
clip_model = CLIPModel()
query_embedding = clip_model.generate_text_embedding(query)

# Load the saved FAISS index
index = faiss.read_index(f"{dataset_name}_image_embeddings.index")

# Assume `query_embedding` is a NumPy array of shape (1, D)
query_embedding_np = query_embedding.cpu().numpy() if torch.is_tensor(query_embedding) else query_embedding

# Search the FAISS index for the top 5 nearest neighbors
k = 5  # Number of nearest neighbors
distances, indices = index.search(query_embedding, k)