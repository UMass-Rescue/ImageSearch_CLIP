from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import FileInput, Input, TextInput

# Text Query
text_query = "Finger licking dessert"
# Path to the image that is used for search
image_path = (
    "./datasets/coco/train/000000000612_jpg.rf.656879428df938a1a000bc255a193ccd.jpg"
)

print("Searching by text...\n")
url = "http://127.0.0.1:5000/search_by_text"  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object

inputs = {"text_query": Input(root=TextInput.model_validate({"text": text_query}))}
parameters = {"dataset_name": "clientExampleDataset", "num_results": 3}
response = client.request(inputs, parameters)  # Send a request to the server
print(response)  # Print the response

print("\n\n------------------------------------------------\n\n")

print("Searching by image...\n")
url = "http://127.0.0.1:5000/search_by_image"  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object

inputs = {"image_path": Input(root=FileInput.model_validate({"path": image_path}))}
parameters = {"dataset_name": "clientExampleDataset", "num_results": 3}
response = client.request(inputs, parameters)  # Send a request to the server
print(response)  # Print the response
