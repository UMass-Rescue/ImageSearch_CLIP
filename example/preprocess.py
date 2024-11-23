from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import DirectoryInput, Input

# Path to the COCO-128 image dataset
# dataset_name = "coco"
# image_dir = './coco/train'  # Update with your image path

# # Disable SSL certificate verification
# ssl._create_default_https_context = ssl._create_unverified_context

# clip_model = CLIPModel()
# clip_model.preprocess_images(image_dir, dataset_name)


url = "http://127.0.0.1:5000/dataset_processing"  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object

inputs = {
    "input_dir": Input(
        root=DirectoryInput.model_validate(
            {"path": "./coco/train"}
        )
    )
}
parameters = {
    "dataset_name": "coco"
}
response = client.request(inputs, parameters)  # Send a request to the server
print(response)  # Print the response
