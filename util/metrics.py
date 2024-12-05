import os
from pathlib import Path

from model.model import CLIPModel
from util.util import is_image_file


def evaluate_image_search(dataset_path, dataset_name):
    """
    Evaluate Top-1, Top-5 Accuracy and Recall for the image search system.

    Args:
        dataset_path (str): Path to the dataset containing all images.
        dataset_name (str): Name of the dataset to pass to the search function.

    Returns:
        dict: A summary of Top-1, Top-5 Accuracy and Recall metrics.
    """
    total_queries = 0
    top_1_correct = 0
    top_5_correct = 0
    total_recall = 0
    model = CLIPModel()

    # List all image files in the dataset directory
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for img_file in files:
            img_path = os.path.join(root, img_file)
            if is_image_file(img_path):
                image_files.append(img_path)

    for image_path in image_files:
        total_queries += 1

        # Get the top num_results similar images
        retrieved_images = model.search_by_image(image_path, dataset_name, 5)
        retrieved_images = [Path(img["result"]).resolve() for img in retrieved_images]

        # Assume each image's nearest match is itself
        relevant_images = {Path(image_path).resolve()}  # Ground truth relevant images
        retrieved_set = set(retrieved_images[:5])

        # Check Top-1 and Top-5 accuracy
        if relevant_images & set(retrieved_images[:1]):
            top_1_correct += 1
        if relevant_images & set(retrieved_images[:5]):
            top_5_correct += 1

        # Calculate Recall
        tp = len(relevant_images & retrieved_set)  # Relevant and retrieved
        fn = len(relevant_images - retrieved_set)  # Relevant but not retrieved

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        total_recall += recall

    # Average metrics across all queries
    avg_recall = total_recall / total_queries if total_queries > 0 else 0
    top_1_accuracy = top_1_correct / total_queries if total_queries > 0 else 0
    top_5_accuracy = top_5_correct / total_queries if total_queries > 0 else 0

    return {
        "total_queries": total_queries,
        "top_1_accuracy": top_1_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "average_recall": avg_recall,
    }
