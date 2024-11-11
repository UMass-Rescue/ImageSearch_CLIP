from typing import TypedDict, List
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import DirectoryInput, ResponseBody, TextResponse, TextInput, FileResponse, BatchFileResponse, FileType, FileInput

from model import CLIPModel

model = CLIPModel()
server = MLServer(__name__)

available_datasets: List[str] = []

class DatasetProcessingInput(TypedDict):
    input_dir: DirectoryInput

class DatasetProcessingParameters(TypedDict):
    dataset_name: str

@server.route("/dataset_processing")
def dataset_processing(inputs: DatasetProcessingInput, parameters: DatasetProcessingParameters) -> ResponseBody:
    available_datasets.append(parameters['dataset_name'])
    directory_path = inputs['input_dir'].path
    result = model.preprocess_images(directory_path, parameters['dataset_name'])
    response = TextResponse(value=result)
    return ResponseBody(root=response)

class TxtInput(TypedDict):
    text_query: TextInput

class SearchParameters(TypedDict):
    dataset_name: str
    num_results: int

@server.route("/search_by_text")
def search_by_text(inputs: TxtInput, parameters: SearchParameters) -> ResponseBody:
    text_query = inputs['text_query'].text
    results = model.search_by_text(text_query, parameters['dataset_name'], parameters['num_results'])
    image_results = [FileResponse(file_type=FileType.IMG, path=res["result"]) for res in results]
    response = BatchFileResponse(files=image_results)
    return ResponseBody(root=response)

class ImageInput(TypedDict):
    image_path: FileInput

@server.route("/search_by_image")
def search_by_image(inputs: ImageInput, parameters: SearchParameters) -> ResponseBody:
    image_path = inputs['image_path'].path
    results = model.search_by_image(image_path, parameters['dataset_name'], parameters['num_results'])
    image_results = [FileResponse(file_type=FileType.IMG, path=res["result"]) for res in results]
    response = BatchFileResponse(files=image_results)
    return ResponseBody(root=response)

server.run()