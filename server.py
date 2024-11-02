from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import DirectoryInput, ResponseBody, TextResponse, TextInput, FileResponse, BatchFileResponse, FileType

from model import CLIPModel

model = CLIPModel()
server = MLServer(__name__)

class DatasetProcessingInput(TypedDict):
    input_dir: DirectoryInput

class DatasetProcessingParameters(TypedDict):
    dataset_name: str

@server.route("/dataset_processing")
def dataset_processing(inputs: DatasetProcessingInput, parameters: DatasetProcessingParameters) -> ResponseBody:
    directory_path = inputs['input_dir'].path
    result = model.preprocess_images(directory_path, parameters['dataset_name'])
    response = TextResponse(value=result)
    return ResponseBody(root=response)

class SearchInput(TypedDict):
    text_query: TextInput

class SearchParameters(TypedDict):
    dataset_name: str

@server.route("/search_by_text")
def search_by_text(inputs: SearchInput, parameters: SearchParameters) -> ResponseBody:
    print("Inside server")
    text_query = inputs['text_query'].text
    print(text_query)
    results = model.search_by_text(text_query, parameters['dataset_name'])
    print(results)
    image_results = [FileResponse(file_type=FileType.IMG, path=res["result"]) for res in results]
    response = BatchFileResponse(files=image_results)
    return ResponseBody(root=response)

server.run()