# ImageSearch_CLIP
This project provides a scalable solution for efficient image search using OpenAI's CLIP model, FAISS, and PostgreSQL. Initially, it processes large image datasets to generate embeddings for each image using the CLIP model. These embeddings are indexed using FAISS for fast similarity searches, while a PostgreSQL database stores mappings of indexes to their file paths.

Once the setup is complete, users can input a query in the form of text or an image. The system generates an embedding for the query using the CLIP model, searches the FAISS index for similar embeddings, and retrieves the corresponding image paths from the PostgreSQL database.

## Setup
**1. Install git and postgres**

[Download git](https://git-scm.com/downloads)


[Download PostgreSQL](https://www.postgresql.org/download/)


**2. Install pipenv and start virtual env**
```
pip install pipenv
```
```
pipenv shell
```
**3. Install dependencies**
```
pipenv install
```
**4. Configure Database**

- Create a .env file in the root directory of your project if it doesn't already exist.

- Add the following properties to the .env file:
  ```
  DB_HOST=localhost
  DB_PORT=5432
  DB_USER=postgres
  DB_PASSWORD=*****
  ```
  **Note**: Make sure to update the DB_HOST, DB_PORT, DB_USER, and DB_PASSWORD values according to your local database setup:
  
    - DB_HOST: Specify the host where your database is running (e.g., localhost for a local setup or the IP address/domain for a remote database).
  
    - DB_PORT: Set the port number for your database (default for PostgreSQL is 5432).
  
    - DB_USER: Enter your database username.
  
    - DB_PASSWORD: Provide the correct password for your database user.

- Save the .env file with the updated values.

## Flask-ML

**Starting server**
```
python -m server.server
```
**Client example**
> Preprocess the dataset first, then perform searches later.

*1. image dataset preprocessing* (Update the path to the test dataset in preprocess.py)
```
python example/preprocess.py
```

*2. test search feature* (Update the text query and the path to the image that you want to use for image search in search.py)
```
python example/search.py
```
## Command line Interface
> Preprocess the dataset first by giving path to the dataset directory (--input_dir) and assign it a dataset name (--name), then later perform search by text (--query) or search by image (--image) using the same dataset name(--name).

**preprocess image dataset**
```
python -m cli.cli_preprocess --input_dir ./dataset --name testDataset
```
```
python -m cli.cli_preprocess -i ./dataset -n testDataset
```

**search by text**
```
python -m cli.cli_search --query "finger licking dessert" --name testDataset
```
```
python -m cli.cli_search -q "finger licking dessert" -n testDataset
```

**search by image**
```
python -m cli.cli_search --image ./test_image.jpg --name testDataset
```
```
python -m cli.cli_search -i ./test_image.jpg -n testDataset
```

**Optional - Specify number of results**
```
python -m cli.cli_search --query "finger licking dessert" --name testDataset --num_results 3
```
```
python -m cli.cli_search -i ./test_image.jpg -n testDataset -k 3
```
> default number of results is 5.

## Model Evaluation
**Evaluation Process**

The evaluation of the Image Search model was conducted using a dataset of images. For each image in the dataset, the model was tasked with retrieving a set of results from the same dataset. The goal was to determine if the input image (query image) appeared in the retrieved results. The retrieved results for each query image were then compared to the query image itself, and the evaluation metrics were calculated based on whether the input image was present in the top results.

**Metrics Calculated**
  - Top-1 Accuracy: Percentage of queries where the input image was the top retrieved result.
  - Top-5 Accuracy: Percentage of queries where the input image appeared within the top 5 retrieved results.
  - Recall: Measures the overall ability of the model to retrieve the input image from the dataset.

**Results**

| **Metric**       | **Value** |  
|-------------------|-----------|  
| Top-1 Accuracy    | 1.0       |  
| Top-5 Accuracy    | 1.0       |  
| Recall            | 1.0       |  

**Steps to reproduce the results**

The script 'cli/cli_evaluation.py' is a command-line interface (CLI) tool to evaluate the Image Search model for a given dataset. This takes as an input, the path to the dataset directory (--input_dir) on which the model is to be evaluated and a dataset name (--name) which acts as an identifier to the dataset.

This script evaluates the top-1 accuracy, top-5 accuracy and recall of the model. 

**Running the Script**
```
python -m cli.cli_evaluation -i <input_directory> -n <dataset_name>

```
**Example**
```
python -m cli.cli_evaluation -i ./dataset -n testDataset

```
