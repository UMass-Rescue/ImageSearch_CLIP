from model import *
from db import *


# Path to the COCO-128 image dataset
dataset_name = "coco"
image_dir = './coco/train'  # Update with your image path

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

clip_model = CLIPModel()

# Load and preprocess all images
image_paths, processed_images = clip_model.load_and_preprocess_images(image_dir)
print(f"Loaded and preprocessed {len(processed_images)} images.")

# Generate embeddings for all preprocessed images
image_embeddings = clip_model.generate_image_embeddings(processed_images)
print(f"Generated embeddings for {image_embeddings.shape[0]} images.")

data_indexing = DataIndexing(dataset_name)
faiss_index = data_indexing.faiss_indexing(image_embeddings)
