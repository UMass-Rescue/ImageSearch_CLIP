import torch
import faiss
import psycopg2
from dotenv import load_dotenv
import os


class DataIndexing:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def _convert_image_embeddings_to_numpy(self, image_embeddings):
        return image_embeddings.cpu().numpy() if torch.is_tensor(image_embeddings) else image_embeddings

    def faiss_indexing(self, image_embeddings):
        image_embeddings = self._convert_image_embeddings_to_numpy(image_embeddings)
        
        # Get the dimensions of the embeddings
        d = image_embeddings.shape[1]  # This should be 512 for CLIP

        # Initialize a FAISS index
        index = faiss.IndexFlatIP(d)  # cosine similarity; for cosine, normalize embeddings first
                                
        # Add the embeddings to the index
        index.add(image_embeddings)

        faiss.write_index(index, f"{self.dataset_name}_image_embeddings.index")

        return index
    
    def create_db_table(self):

        self._create_database_if_not_exists()

        # Create a connection to the PostgreSQL database
        conn = psycopg2.connect(host=self.DB_HOST, port=self.DB_PORT, dbname=self.DB_NAME, user=self.DB_USER, password=self.DB_PASSWORD)
        cursor = conn.cursor()

        # Dynamically create a table for storing image metadata for a specific dataset
        table_name = f"metadata_{self.dataset_name}"  # Use dataset-specific table name
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                image_index INTEGER,
                image_path TEXT
            )
        ''')

        conn.commit()
        cursor.close()
        conn.close()

    def insert_metadata(self, image_paths):
        # Insert image metadata into a specific dataset's table
        conn = psycopg2.connect(host=self.DB_HOST, port=self.DB_PORT, dbname=self.DB_NAME, user=self.DB_USER, password=self.DB_PASSWORD)
        cursor = conn.cursor()

        table_name = f"metadata_{self.dataset_name}"  # Dataset-specific table
        for idx, path in enumerate(image_paths):
            cursor.execute(f'''
                INSERT INTO {table_name} (image_index, image_path)
                VALUES (%s, %s)
            ''', (idx, path))

        conn.commit()
        cursor.close()
        conn.close()
    