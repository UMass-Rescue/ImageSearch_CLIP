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
### Steps to Use the Rescue Box Application with the Flask-ML Server

1. **Launch the Rescue Box Application**  
   - Download the Rescue Box Application [here](https://github.com/UMass-Rescue/RescueBox-Desktop/releases).
   - Open the application on your system.

2. **Register the Server**  
   - In the Rescue Box application, navigate to the *Register a Model* section.  
   - Enter the following details:  
     - **IP Address:** `127.0.0.1`  
     - **Port Number:** `5000`  
   - Click the *Connect* button to connect the application to the locally running Flask-ML server.

3. **Select a Job**  
   - Once the server is registered, choose from one of the three available jobs:  
     - **Ingest Dataset into the Model (Data Preprocessing):**  
       - Provide the dataset directory path and dataset name.  
       - The server will process the images, create embeddings, and store them in the FAISS index.  
     - **Search Images by Text Query:**  
       - Enter a descriptive text query and select the dataset name from the dropdown.  
       - The server will search for similar images based on the text query.  
     - **Search Images by Image File:**  
       - Upload an image file and select the dataset name.  
       - The server will find visually similar images from the dataset.

4. **View Results**  
   - After running the selected job, the application will display the results (e.g., similar images or processing status).  

5. **Close the Application**  
   - When done, you can disconnect from the server and close the Rescue Box application.
   
<img width="500" alt="image" src="https://github.com/user-attachments/assets/f486cc56-4e64-4884-a465-b6ac128b7cfb">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/94600da7-4cb1-4009-934a-ed82e76a0685">


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

## Model Performance
- On average, the preprocessing step handles *26.56* images per second.
- The average time to process a search request is *0.18* seconds, regardless of the number of results requested or the dataset size, showing minimal impact from these factors.

## Model Evaluation
**1. Evaluation for a custom dataset**

We used a custom dataset (evaluation/Evaluation dataset) of 110 images collected from 10 different categories and preprocessed the model on training set (evaluation/Evaluation dataset/train) consisting of 100 images, 10 from each category and validated the results by performing search using the validation set (evaluation/Evaluation dataset/val) consiting of 10 images, one from each category.

**Metrics Calculated**
  - Precision@1: Measures the proportion of times the first result returned by the model is relevant.
  - Precision@5: Measures the proportion of relevant images in the top 5 results retrieved by the model.

**Results**

| **Metric**       | **Value** |  
|-------------------|-----------|  
| Precision@1            | 1.0       |  
| Precision@5    | 1.0       |  

**Steps to reproduce the results**

Run the script evaluation/evaluate.py
```
python -m evaluation.evaluate

```

**2. Evaluation for a user-given dataset**

The evaluation of the Image Search model was conducted using a dataset of images. For each image in the dataset, the model was tasked with retrieving a set of results from the same dataset. The goal was to determine if the input image (query image) appeared in the retrieved results. The retrieved results for each query image were then compared to the query image itself, and the evaluation metrics were calculated based on whether the input image was present in the top results.

**Metrics Calculated**
  - Hist@1: Percentage of queries where the input image was the top retrieved result.
  - Hits@5: Percentage of queries where the input image appeared within the top 5 retrieved results.
  - Recall: Measures the overall ability of the model to retrieve the input image from the dataset.

**Results**

| **Metric**       | **Value** |  
|-------------------|-----------|  
| Hits@1            | 1.0       |  
| Hits@5    | 1.0       |  
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
