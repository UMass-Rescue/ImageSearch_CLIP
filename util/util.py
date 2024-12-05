from PIL import Image


def dataset_storage_name(dataset_name):
    return f"{dataset_name}_dataset"


def is_valid_dataset_name(dataset_name):
    return dataset_name.isalnum() and " " not in dataset_name


def is_image_file(file_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")

    # Check if the file has a valid image extension
    if not file_path.lower().endswith(valid_extensions):
        return False

    try:
        # Open the file to verify it's a valid image
        with Image.open(file_path) as img:
            img.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, SyntaxError):
        return False
