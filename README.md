# ImageSearch_CLIP
This project provides a scalable solution for efficient image search using OpenAI's CLIP model, FAISS, and PostgreSQL. Initially, it processes large image datasets (e.g., CIFAR-10) to generate embeddings for each image using the CLIP model. These embeddings are indexed using FAISS for fast similarity searches, while a PostgreSQL database stores mappings of image embeddings to their file paths.

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
