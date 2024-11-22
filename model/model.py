import clip
import torch
import os
from PIL import Image
from database.faiss_db import FAISSDatabase
from database.psql_db import PSQLDatabase

class CLIPModel:
    def __init__(self):
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Databases
        self.faiss_db = FAISSDatabase()
        self.psql_db = PSQLDatabase()

    # Function to load and preprocess images using CLIP
    def _load_and_preprocess_images(self, image_dir):
        image_paths = []
        processed_images = []
        
        for root, _, files in os.walk(image_dir):
            for img_file in files:
                if img_file.endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(image_dir, img_file)
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
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)  # Normalize embeddings
            
        return image_embeddings
    
    # Function to generate text embeddings for a given query

    def _generate_text_embedding(self, query):
        text = clip.tokenize([query]).to(self.device)  # Tokenize the input query
        with torch.no_grad():
            text_embedding = self.model.encode_text(text)  # Generate text embeddings
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
        return text_embedding

    def _generate_image_embedding_from_input(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)  # type: ignore # Preprocess and move to device (GPU or CPU)
        
        # Generate embedding
        with torch.no_grad():
            image_embedding = self.model.encode_image(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
        
        return image_embedding
    
    def _save_to_db(self, dataset_name, image_embeddings, image_paths):
        # Write Faiss index
        self.faiss_db.write_to_faiss(dataset_name, image_embeddings)

        # Insert image paths into the psql table
        self.psql_db.store_image_paths(dataset_name, image_paths)
    
    def preprocess_images(self, image_dir: str, dataset_name: str):
        # Load and preprocess all images
        image_paths, processed_images = self._load_and_preprocess_images(image_dir)
        print(f"Loaded and preprocessed {len(processed_images)} images.")

        # Generate embeddings for all preprocessed images
        image_embeddings = self._generate_image_embeddings(processed_images)
        print(f"Generated embeddings for {image_embeddings.shape[0]} images.")

        self._save_to_db(dataset_name, image_embeddings, image_paths)
        print(f"Embeddings saved to database")

        return f"{dataset_name} dataset processed successfully!!"
    
    def _search(self, dataset_name, embedding, k):
        top_k_distances, top_k_indices = self.faiss_db.search_faiss(dataset_name, embedding, k)

        images_data = self.psql_db.fetch_image_paths(dataset_name, top_k_indices)

        # Combine FAISS results with metadata and display the results
        top_n_results = [(images_data[i], top_k_distances[j]) for j, i in enumerate(top_k_indices) if i in images_data]
        results = []
        # Display results
        for i, (file_path, score) in enumerate(top_n_results):
            print(f"Result {i+1}: {file_path} (Distance: {score:.4f})")
            results.append({'result': file_path, 'title': os.path.splitext(os.path.basename(file_path))[0]})
        
        return results

    def search_by_text(self, query: str, dataset_name: str, k: int):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        query_embedding = self._generate_text_embedding(query)
        return self._search(dataset_name, query_embedding, k)
        
    def search_by_image(self, image_path: str, dataset_name: str, k: int):
        image_embedding = self._generate_image_embedding_from_input(image_path)
        return self._search(dataset_name, image_embedding, k)
