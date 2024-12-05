from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import DirectoryInput, Input

# Path to the test dataset
dataset_path = './datasets/coco/'

url = "http://127.0.0.1:5000/dataset_processing"  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object

inputs = {
    "input_dir": Input(
        root=DirectoryInput.model_validate(
            {"path": dataset_path}
        )
    )
}
parameters = {
    "dataset_name": "clientExampleDataset"
}
response = client.request(inputs, parameters)  # Send a request to the server
print(response)  # Print the response
