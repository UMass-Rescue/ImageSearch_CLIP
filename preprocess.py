from model import *

# Path to the COCO-128 image dataset
dataset_name = "coco"
image_dir = './coco/train'  # Update with your image path

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

clip_model = CLIPModel()
clip_model.preprocess_images(image_dir, dataset_name)
