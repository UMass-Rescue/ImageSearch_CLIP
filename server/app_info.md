# Image Search CLIP

**Welcome!**

This project provides a robust and scalable solution for efficient image search by combining OpenAI's CLIP model, FAISS (Facebook AI Similarity Search), and PostgreSQL. It allows users to preprocess large image datasets, generate embeddings, and store them for fast similarity searches. Once the setup is complete, users can perform searches using text or image queries to retrieve relevant image paths from the dataset.

The system is designed to handle large-scale datasets and leverages the power of machine learning for embedding generation and FAISS for high-speed similarity search.

## How to Use the App

### Step 1: Preprocess the Dataset

Before you can perform searches, the dataset must be preprocessed.

- **Launch Preprocessing**:  
   Use the app to send a request with the following details:  
   - **Image Directory Path**: The location where your images are stored.  
   - **Dataset Name**: A unique name to identify your dataset during future searches.

- The app will process the images, generate searchable indexes, and store the results. 

   > **Note**: Preprocessing can take time depending on the dataset size and compute resources. Ensure you allocate sufficient resources before starting.

---

### Step 2: Perform a Search

After preprocessing, you can perform searches on the dataset.

- **Submit a Search Query**:  
   Provide the following details in the search request:
   - **Query Type**: Either a text description or an image file.
   - **Dataset Name**: The same name used during preprocessing.
   - **Number of Results**: Specify how many top results you want.

- **Output**:  
   The app will search the indexed data and return paths to the most relevant images.

## Example Use Cases

- **Preprocessing a Dataset**:  
   - Input: Image directory and dataset name.  
   - Example:  
     - `Image Directory`: `/data/images/`  
     - `Dataset Name`: `wildlife_photos`

- **Performing a Search**:  
   - Query Input: Text or image.  
   - Example:  
     - `Query Type`: Text - "A lion in the grasslands"  
     - `Dataset Name`: `wildlife_photos`  
     - `Number of Results`: 5

## Support

If you encounter any issues, have questions, or wish to suggest features, please raise an issue on our [GitHub repository](https://github.com/UMass-Rescue/ImageSearch_CLIP). 

Thank you for using the Scalable Image Search App!
