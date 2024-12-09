import os
from pathlib import Path

from database.psql import PSQLDatabase
from model.model import CLIPModel
from util.util import is_image_file

model = CLIPModel()
db = PSQLDatabase()

# Preprocessing the datatset
dataset_name = "evaluationDataset"
train_dataset_path = "./evaluation/Evaluation dataset/train"

if dataset_name not in db.get_all_datasets():
    model.preprocess_images(train_dataset_path, dataset_name)

# Validating the model
val_dataset_path = "./evaluation/Evaluation dataset/val"

# List all image files in the dataset directory
image_files = []
for root, _, files in os.walk(val_dataset_path):
    for img_file in files:
        img_path = os.path.join(root, img_file)
        if is_image_file(img_path):
            image_files.append(img_path)

# Initialize metrics
precision_at_1_total = 0
precision_at_5_total = 0
num_queries = 0

for image_path in image_files:
    # Extract the base filename from the image path
    base_filename = Path(image_path).stem

    # Collect all training images whose filenames start with the base filename
    relevant_images = {
        Path(os.path.join(root, train_file)).resolve()
        for root, _, files in os.walk(train_dataset_path)
        for train_file in files
        if Path(train_file).stem.startswith(base_filename)
        and is_image_file(os.path.join(root, train_file))
    }

    retrieved_images = model.search_by_image(image_path, dataset_name, 5)
    retrieved_images = [Path(img["result"]).resolve() for img in retrieved_images]

    retrieved_set = set(retrieved_images[:5])

    # Calculate Precision@1
    if retrieved_images[0] in relevant_images:
        precision_at_1_total += 1

    # Calculate Precision@5
    relevant_in_top_5 = len(set(retrieved_images[:5]) & relevant_images)
    precision_at_5_total += relevant_in_top_5 / 5

    # Increment query count
    num_queries += 1

# Compute final metrics
precision_at_1 = precision_at_1_total / num_queries
precision_at_5 = precision_at_5_total / num_queries

print(f"Precision@1: {precision_at_1:.4f}")
print(f"Precision@5: {precision_at_5:.4f}")
