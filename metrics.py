import os
from model.model import CLIPModel
from pathlib import Path

def evaluate_image_search(dataset_path, dataset_name):
    """
    Evaluate Top-1, Top-5 Accuracy, Precision, and Recall for the image search system.

    Args:
        dataset_path (str): Path to the dataset containing all images.
        dataset_name (str): Name of the dataset to pass to the search function.
        num_results (int): Maximum number of similar images to retrieve.
        search_function (callable): Function to perform the search (e.g., search_by_image).

    Returns:
        dict: A summary of Top-1, Top-5 Accuracy, Precision, and Recall metrics.
    """
    total_queries = 0
    top_1_correct = 0
    top_5_correct = 0
    total_precision = 0
    total_recall = 0
    model = CLIPModel()

    # List all image files in the dataset
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('jpg', 'png', 'jpeg'))]

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

        # Calculate Precision and Recall
        tp = len(relevant_images & retrieved_set)  # Relevant and retrieved
        fp = len(retrieved_set - relevant_images)  # Retrieved but not relevant
        fn = len(relevant_images - retrieved_set)  # Relevant but not retrieved

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Accumulate metrics
        # total_precision += precision
        total_recall += recall

    # Average metrics across all queries
    avg_precision = total_precision / total_queries if total_queries > 0 else 0
    avg_recall = total_recall / total_queries if total_queries > 0 else 0
    top_1_accuracy = top_1_correct / total_queries if total_queries > 0 else 0
    top_5_accuracy = top_5_correct / total_queries if total_queries > 0 else 0

    return {
        "total_queries": total_queries,
        "top_1_accuracy": top_1_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "average_recall": avg_recall,
    }

