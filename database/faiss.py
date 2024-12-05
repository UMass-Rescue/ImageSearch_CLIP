import os

import faiss
import numpy as np
import torch

from util.util import dataset_storage_name


class FAISSDatabase:
    def __init__(self):
        self.default_num_results = 5
        self.output_dir = "faiss_index"
        os.makedirs(self.output_dir, exist_ok=True)

    def _convert_image_embeddings_to_numpy(self, image_embeddings):
        return (
            image_embeddings.cpu().numpy()
            if torch.is_tensor(image_embeddings)
            else image_embeddings
        )

    def write_to_faiss(self, dataset_name, image_embeddings):
        image_embeddings = self._convert_image_embeddings_to_numpy(image_embeddings)

        # Normalize embeddings (L2 normalization)
        image_embeddings = image_embeddings / np.linalg.norm(
            image_embeddings, axis=1, keepdims=True
        )

        # Get the dimensions of the embeddings
        d = image_embeddings.shape[1]  # This should be 512 for CLIP

        # Initialize a FAISS index
        index = faiss.IndexFlatIP(
            d
        )  # cosine similarity; for cosine, normalize embeddings first

        # Add the embeddings to the index
        index.add(image_embeddings)

        index_file_path = os.path.join(
            self.output_dir, f"{dataset_storage_name(dataset_name)}.index"
        )
        faiss.write_index(index, index_file_path)

    def search_faiss(self, dataset_name, query_embedding, k):
        # Ensure the embedding is on the CPU before passing to FAISS
        query_embedding = query_embedding.cpu()

        # Load the saved FAISS index
        index_file_path = os.path.join(
            self.output_dir, f"{dataset_storage_name(dataset_name)}.index"
        )
        index = faiss.read_index(index_file_path)

        # Search the FAISS index for the top k nearest neighbors
        k = k or self.default_num_results
        distances, indices = index.search(query_embedding, k)

        # Convert the FAISS indices array to a list for use in SQL queries
        top_k_indices = indices[0].tolist()
        top_k_distances = distances[0].tolist()

        return top_k_distances, top_k_indices
