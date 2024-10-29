from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import TextInput, Input

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Main script for searching FAISS index
# dataset_name = "coco"  # Example dataset name
# query = "man wearing red shirt"

# # Generate the query embedding
# clip_model = CLIPModel()

# clip_model.search(query, dataset_name)

url = "http://127.0.0.1:5000/search_by_text"  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object

inputs = {
    "text_query": Input(
        root=TextInput.model_validate(
            {"text": "man in red shirt"}
        )
    )
}
parameters = {
    "dataset_name": "coco"
}
response = client.request(inputs, parameters)  # Send a request to the server
print(response)  # Print the response
