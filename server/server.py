from typing import TypedDict, List
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    TaskSchema, DirectoryInput, TextInput, FileType, FileInput, InputSchema, InputType, 
    ParameterSchema, TextParameterDescriptor, EnumParameterDescriptor, EnumVal, IntParameterDescriptor, 
    ResponseBody, TextResponse, FileResponse, BatchFileResponse)

from model.model import CLIPModel
from database.psql import PSQLDatabase

model = CLIPModel()
db = PSQLDatabase()
server = MLServer(__name__)
server.add_app_metadata(
    name="Image Search using CLIP",
    author="Sahithi Singireddy, Sravani Gona",
    version="0.1.0",
    info=load_file_as_string("README.md"),
)

available_datasets: List[str] = []

def get_dataset_processing_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="input_dir",
                label="Dataset Directory",
                input_type=InputType.DIRECTORY,
            )
        ],
        parameters=[
            ParameterSchema(
                key="dataset_name",
                label="DataSet Name",
                value=TextParameterDescriptor(default=""),
            ),
        ],
    )

class DatasetProcessingInput(TypedDict):
    input_dir: DirectoryInput

class DatasetProcessingParameters(TypedDict):
    dataset_name: str

@server.route(
        "/dataset_processing",
        order=0,
        short_title="Ingest Dataset into the model",
        task_schema_func=get_dataset_processing_task_schema)
def dataset_processing(inputs: DatasetProcessingInput, parameters: DatasetProcessingParameters) -> ResponseBody:
    dataset_name = parameters["dataset_name"]

    if dataset_name in db.get_all_datasets():
        raise ValueError("Dataset name already exists.")
    
    directory_path = inputs['input_dir'].path
    result = model.preprocess_images(directory_path, parameters['dataset_name'])
    response = TextResponse(value=result)
    return ResponseBody(root=response)

def get_search_by_text_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="text_query",
                label="Text Query",
                input_type=InputType.TEXT,
            )
        ],
        parameters=[
            ParameterSchema(
                key="dataset_name",
                label="DataSet Name",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key=dataset_name, label=dataset_name)
                        for dataset_name in db.get_all_datasets()
                    ],
                    message_when_empty="No datasets found",
                    default=available_datasets[0] if len(available_datasets) > 0 else "",
                ),
            ),
            ParameterSchema(
                key="num_results",
                label="Number of Results",
                value=IntParameterDescriptor(default=5),
            ),
        ],
    )

class TxtInput(TypedDict):
    text_query: TextInput

class SearchParameters(TypedDict):
    dataset_name: str
    num_results: int

@server.route(
        "/search_by_text",
        order=1,
        short_title="Search Images by Text Query",
        task_schema_func=get_search_by_text_task_schema)
def search_by_text(inputs: TxtInput, parameters: SearchParameters) -> ResponseBody:
    text_query = inputs['text_query'].text
    results = model.search_by_text(text_query, parameters['dataset_name'], parameters['num_results'])
    image_results = [FileResponse(title=res["title"], file_type=FileType.IMG, path=res["result"]) for res in results]
    response = BatchFileResponse(files=image_results)
    return ResponseBody(root=response)

def get_search_by_image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="image_path",
                label="Image Path",
                input_type=InputType.FILE,
            )
        ],
        parameters=[
            ParameterSchema(
                key="dataset_name",
                label="DataSet Name",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key=dataset_name, label=dataset_name)
                        for dataset_name in db.get_all_datasets()
                    ],
                    message_when_empty="No datasets found",
                    default=available_datasets[0] if len(available_datasets) > 0 else "",
                ),
            ),
            ParameterSchema(
                key="num_results",
                label="Number of Results",
                value=IntParameterDescriptor(default=5),
            ),
        ],
    )

class ImageInput(TypedDict):
    image_path: FileInput

@server.route(
        "/search_by_image",
        order=2,
        short_title="Search Images by Image File",
        task_schema_func=get_search_by_image_task_schema)
def search_by_image(inputs: ImageInput, parameters: SearchParameters) -> ResponseBody:
    image_path = inputs['image_path'].path
    results = model.search_by_image(image_path, parameters['dataset_name'], parameters['num_results'])
    image_results = [FileResponse(title=res["title"], file_type=FileType.IMG, path=res["result"]) for res in results]
    response = BatchFileResponse(files=image_results)
    return ResponseBody(root=response)

server.run()