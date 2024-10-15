import torch
import faiss
import numpy as np

def convert_image_embeddings_to_numpy(image_embeddings):
    return image_embeddings.cpu().numpy() if torch.is_tensor(image_embeddings) else image_embeddings

def faiss_indexing(image_embeddings):
    image_embeddings = convert_image_embeddings_to_numpy(image_embeddings)
    # Get the dimensions of the embeddings
    d = image_embeddings.shape[1]  # This should be 512 for CLIP

    # Initialize a FAISS index
    index = faiss.IndexFlatIP(d)  # cosine similarity; for cosine, normalize embeddings first
                            
    # Add the embeddings to the index
    index.add(image_embeddings)

    faiss.write_index(index, "image_embeddings.index")

    return index