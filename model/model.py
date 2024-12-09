import os
from time import perf_counter

import clip
import torch
from PIL import Image

from database.faiss import FAISSDatabase
from database.psql import PSQLDatabase
from util.util import is_image_file, is_valid_dataset_name


class CLIPModel:
    def __init__(self):
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Databases
        self.faiss_db = FAISSDatabase()
        self.psql_db = PSQLDatabase()
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Function to load and preprocess images using CLIP
    def _load_and_preprocess_images(self, image_dir):
        image_paths = []
        processed_images = []

        for root, _, files in os.walk(image_dir):
            for img_file in files:
                img_path = os.path.join(root, img_file)
                if is_image_file(img_path):
                    image_paths.append(img_path)

                    # Load and preprocess the image
                    image = Image.open(img_path).convert("RGB")
                    image = self.preprocess(image).unsqueeze(0).to(self.device)  # type: ignore # Preprocess and move to device (GPU or CPU)
                    processed_images.append(image)

        return image_paths, processed_images

    # Function to generate image embeddings
    def _generate_image_embeddings(self, processed_images):
        with torch.no_grad():
            image_embeddings = []
            for image in processed_images:
                image_feature = self.model.encode_image(image)  # Generate embeddings
                image_embeddings.append(image_feature)

            # Concatenate all embeddings into a single tensor
            image_embeddings = torch.cat(image_embeddings, dim=0)
            image_embeddings /= image_embeddings.norm(
                dim=-1, keepdim=True
            )  # Normalize embeddings

        return image_embeddings

    # Function to generate text embeddings for a given query
    def _generate_text_embedding(self, query):
        text = clip.tokenize([query]).to(self.device)  # Tokenize the input query
        with torch.no_grad():
            text_embedding = self.model.encode_text(text)  # Generate text embeddings
            text_embedding /= text_embedding.norm(
                dim=-1, keepdim=True
            )  # Normalize the embedding
        return text_embedding

    def _generate_image_embedding_from_input(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)  # type: ignore # Preprocess and move to device (GPU or CPU)

        # Generate embedding
        with torch.no_grad():
            image_embedding = self.model.encode_image(image)
            image_embedding /= image_embedding.norm(
                dim=-1, keepdim=True
            )  # Normalize the embedding

        return image_embedding

    def _save_to_db(self, dataset_name, dataset_size, image_embeddings, image_paths):
        # Write Faiss index
        self.faiss_db.write_to_faiss(dataset_name, image_embeddings)

        # Insert image paths into the psql table
        self.psql_db.store_image_paths(dataset_name, dataset_size, image_paths)

    def preprocess_images(self, image_dir: str, dataset_name: str):
        start_time = perf_counter()
        if not is_valid_dataset_name(dataset_name):
            raise ValueError(
                "Dataset name must be a single alphanumeric word without spaces or special characters"
            )

        # Load and preprocess all images
        image_paths, processed_images = self._load_and_preprocess_images(image_dir)
        if len(processed_images) == 0:
            raise ValueError("Empty directory, no images found.")
        print(f"Loaded and preprocessed {len(processed_images)} images.")

        # Generate embeddings for all preprocessed images
        image_embeddings = self._generate_image_embeddings(processed_images)
        dataset_size = image_embeddings.shape[0]
        print(f"Generated embeddings for {dataset_size} images.")

        self._save_to_db(dataset_name, dataset_size, image_embeddings, image_paths)
        print(f"Embeddings saved to database")

        end_time = perf_counter()
        execution_time = end_time - start_time
        speed_text = f"Processing time: {execution_time:.2f} sec, speed: {(dataset_size/execution_time):.2f} images/sec"
        print(speed_text)
        return f"{dataset_name} dataset with {dataset_size} images processed successfully!! \n {speed_text}"

    def _search(self, dataset_name, embedding, k):
        if k > self.psql_db.get_dataset_size(dataset_name):
            raise ValueError("num results requested is larger than datset size")

        top_k_distances, top_k_indices = self.faiss_db.search_faiss(
            dataset_name, embedding, k
        )

        images_data = self.psql_db.fetch_image_paths(dataset_name, top_k_indices)

        # Combine FAISS results with metadata and display the results
        top_k_results = [
            (images_data[i], top_k_distances[j])
            for j, i in enumerate(top_k_indices)
            if i in images_data
        ]
        results = []
        # Display results
        for i, (file_path, score) in enumerate(top_k_results):
            results.append(
                {
                    "result": file_path,
                    "title": os.path.splitext(os.path.basename(file_path))[0],
                }
            )

        return results

    def search_by_text(self, query: str, dataset_name: str, k: int):
        if query is None or query.strip() == "":
            raise ValueError("Invalid input text query.")

        start_time = perf_counter()
        query_embedding = self._generate_text_embedding(query)
        result = self._search(dataset_name, query_embedding, k)

        end_time = perf_counter()
        execution_time = end_time - start_time
        print(
            f"Processing time: {execution_time:.2f} sec, speed: {(k/execution_time):.2f} images/sec"
        )
        return result

    def search_by_image(self, image_path: str, dataset_name: str, k: int):
        if not is_image_file(image_path):
            raise ValueError("Invalid or corrupted input image.")

        start_time = perf_counter()
        image_embedding = self._generate_image_embedding_from_input(image_path)
        result = self._search(dataset_name, image_embedding, k)
        end_time = perf_counter()
        execution_time = end_time - start_time
        print(
            f"Processing time: {execution_time:.2f} sec, , speed: {(k/execution_time):.2f} images/sec"
        )
        return result
