import ssl
import clip
import torch
import os
from PIL import Image

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("CLIP model loaded successfully!")



# Path to the COCO-128 image dataset
image_dir = './coco/train'  # Update with your image path

# Function to load and preprocess images using CLIP
def load_and_preprocess_images(image_dir, preprocess):
    image_paths = []
    processed_images = []
    
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(image_dir, img_file)
            image_paths.append(img_path)
            
            # Load and preprocess the image
            image = Image.open(img_path).convert("RGB")
            image = preprocess(image).unsqueeze(0).to(device)  # Preprocess and move to device (GPU or CPU)
            processed_images.append(image)
    
    return image_paths, processed_images

# Load and preprocess all images
image_paths, processed_images = load_and_preprocess_images(image_dir, preprocess)
print(f"Loaded and preprocessed {len(processed_images)} images.")

# Function to generate image embeddings
def generate_image_embeddings(processed_images, model):
    with torch.no_grad():
        image_embeddings = []
        for image in processed_images:
            image_feature = model.encode_image(image)  # Generate embeddings
            image_embeddings.append(image_feature)
        
        # Concatenate all embeddings into a single tensor
        image_embeddings = torch.cat(image_embeddings, dim=0)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)  # Normalize embeddings
        
    return image_embeddings

# Generate embeddings for all preprocessed images
image_embeddings = generate_image_embeddings(processed_images, model)
print(f"Generated embeddings for {image_embeddings.shape[0]} images.")
