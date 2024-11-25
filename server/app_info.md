# Image Search CLIP

**Welcome!**

This project provides a robust and scalable solution for efficient image search by combining OpenAI's CLIP model, FAISS (Facebook AI Similarity Search), and PostgreSQL. It allows users to preprocess large image datasets, generate embeddings, and store them for fast similarity searches. Once the setup is complete, users can perform searches using text or image queries to retrieve relevant image paths from the dataset.

The system is designed to handle large-scale datasets and leverages the power of machine learning for embedding generation and FAISS for high-speed similarity search.

# How to Use the App

## Step 1: Preprocess the Dataset

Before you can perform searches, the dataset must be preprocessed.

- **Launch Preprocessing**:  
   Use the app to send a request with the following details:  
   - **Image Directory Path**: The location where your images are stored.  
   - **Dataset Name**: A unique name to identify your dataset during future searches.

- The app will process the images, generate searchable indexes, and store the results. 

   > **Note**: Preprocessing can take time depending on the dataset size and compute resources. Ensure you allocate sufficient resources before starting.

## Step 2: Perform a Search

After preprocessing your dataset, you can perform searches using one of two endpoints: **search by text** or **search by image**.

### **Search by Text**
Use this endpoint to search the dataset using a descriptive text query.

**Input Details**:
- **Query Type**: Provide a text description. For example, *"A cat sitting on a couch"*.
- **Dataset Name**: Specify the name of the dataset used during preprocessing.
- **Number of Results**: Define how many top matches to return.

### **Search by Image**
Use this endpoint to search the dataset by providing an image as a query.

**Input Details**:
- **Query Type**: An image file path to serve as the query.
- **Dataset Name**: Specify the dataset name used during preprocessing.
- **Number of Results**: Define the number of top matches to return.


**Output for Both Endpoints**:  
The app compares the query against the dataset embeddings, performs a similarity search, and returns the file paths of the top-matching images from the original dataset.


## Example Use Cases

- **Preprocessing a Dataset**:  
   - Input: Image directory and dataset name.  
   - Example:  
     - `Image Directory`: `/data/images/`  
     - `Dataset Name`: `wildlife_photos`

- **Performing a Search by text**:  
   - Query Input: Text  
   - Example:  
     - `Query`: "A lion in the grasslands"  
     - `Dataset Name`: `wildlife_photos`  
     - `Number of Results`: 3
  - Output:
     - Paths to the 3 most relevant images matching the description.
       
- **Performing a Search by image**:  
   - Query Input: Image
   - Example: 
     - `Query`: `images/query.jpg`
     - `Dataset Name`: `wildlife_photos`
   - Output:
     - Paths to the 5 (default) most relevant images matching the description.

## Support

If you encounter any issues, have questions, or wish to suggest features, please raise an issue on our [GitHub repository](https://github.com/UMass-Rescue/ImageSearch_CLIP). 

Thank you for using the Image Search App!
