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

# Convert the FAISS indices array to a list for use in SQL queries
top_n_indices = indices[0].tolist()  # List of top n indices returned by FAISS

# Fetch the image paths for the top-n returned indices from the PostgreSQL database
data_handler = DataIndexing(dataset_name)
metadata = data_handler.fetch_metadata_by_indices(top_n_indices)

# Combine FAISS results with metadata and display the results
top_n_results = [(metadata[i], distances[0][j]) for j, i in enumerate(top_n_indices) if i in metadata]

# Display results
for i, (path, score) in enumerate(top_n_results):
    print(f"Result {i+1}: {path} (Distance: {score:.4f})")