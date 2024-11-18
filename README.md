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
python server.py
```
**Client example**
> Preprocess the dataset first, then perform searches later.

*1. image dataset preprocessing*
```
python preprocess.py
```

*2. test search feature*
```
python search.py
```
## Command line Interface
> Preprocess the dataset first giving it a name, then perform searches later using the same dataset name.

**preprocess image dataset**
```
python cli_preprocess.py --input_dir ./images --name coco
```
```
python cli_preprocess.py -i ./images -n coco
```

**search by text**
```
python cli_search.py --query "man in red shirt" --name coco
```
```
python cli_search.py -q "man in red shirt" -n coco
```

**search by image**
```
python cli_search.py --image ./image.jpg --name coco
```
```
python cli_search.py -i ./image.jpg -n coco
```

**Optional - Specify number of results**
```
python cli_search.py --query "man in red shirt" --name coco --num_results 3
```
```
python cli_search.py -i ./image.jpg -n coco -k 3
```
> default number of results is 5.
