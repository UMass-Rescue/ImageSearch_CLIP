import torch
import faiss
from dotenv import load_dotenv
import os


class DataIndexing:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def _convert_image_embeddings_to_numpy(self, image_embeddings):
        return image_embeddings.cpu().numpy() if torch.is_tensor(image_embeddings) else image_embeddings

    def faiss_indexing(self, image_embeddings):
        image_embeddings = self._convert_image_embeddings_to_numpy(image_embeddings)
        
        # Get the dimensions of the embeddings
        d = image_embeddings.shape[1]  # This should be 512 for CLIP

        # Initialize a FAISS index
        index = faiss.IndexFlatIP(d)  # cosine similarity; for cosine, normalize embeddings first
                                
        # Add the embeddings to the index
        index.add(image_embeddings)

        faiss.write_index(index, f"{self.dataset_name}_image_embeddings.index")

        return index
    
    