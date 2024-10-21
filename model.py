import ssl
import clip
import torch
import os
from PIL import Image

class CLIPModel:
    def __init__(self):
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    # Function to load and preprocess images using CLIP
    def load_and_preprocess_images(self, image_dir):
        image_paths = []
        processed_images = []
        
        for img_file in os.listdir(image_dir):
            if img_file.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(image_dir, img_file)
                image_paths.append(img_path)
                
                # Load and preprocess the image
                image = Image.open(img_path).convert("RGB")
                image = self.preprocess(image).unsqueeze(0).to(self.device)  # Preprocess and move to device (GPU or CPU)
                processed_images.append(image)
        
        return image_paths, processed_images

    # Function to generate image embeddings
    def generate_image_embeddings(self, processed_images):
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

    def generate_text_embedding(self, query):
        text = clip.tokenize([query]).to(self.device)  # Tokenize the input query
        with torch.no_grad():
            text_embedding = self.model.encode_text(text)  # Generate text embeddings
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
        return text_embedding
