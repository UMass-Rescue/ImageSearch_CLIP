import torch
import faiss
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import os


class DataIndexing:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    # Load environment variables from the .env file
        load_dotenv()

        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = os.getenv("DB_PORT", "5432")
        self.DB_NAME = os.getenv("DB_NAME", "image_search")
        self.DB_USER = os.getenv("DB_USER")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD")

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
    
    def _create_database_if_not_exists(self):
        """
        Create the database if it does not exist.
        """
        # Connect to the default database 'postgres' to check for the target database
        conn = psycopg2.connect(dbname='postgres', user=self.DB_USER, password=self.DB_PASSWORD, host=self.DB_HOST, port=self.DB_PORT)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)  # Allows creating a new database
        
        cursor = conn.cursor()

        # Check if the database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.DB_NAME}'")
        exists = cursor.fetchone()

        if not exists:
            # If the database does not exist, create it
            cursor.execute(f"CREATE DATABASE {self.DB_NAME}")
            print(f"Database '{self.DB_NAME}' created successfully.")
        else:
            print(f"Database '{self.DB_NAME}' already exists.")

        cursor.close()
        conn.close()
    
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
    
    def fetch_metadata_by_indices(self, indices):
        # Fetch image paths from the dataset-specific table based on indices
        conn = psycopg2.connect(host=self.DB_HOST, port=self.DB_PORT, dbname=self.DB_NAME, user=self.DB_USER, password=self.DB_PASSWORD)
        cursor = conn.cursor()

        table_name = f"metadata_{self.dataset_name}"  # Dataset-specific table
        # Create a string of placeholders for the number of indices
        placeholders = ', '.join(['%s'] * len(indices))

        # Construct the SQL query with the placeholders
        query = f'''
            SELECT image_index, image_path 
            FROM metadata_coco 
            WHERE image_index IN ({placeholders})
        '''

        # Execute the query with the list of indices
        cursor.execute(query, indices)  # Pass the list directly
        
        # Fetch the results (returns list of tuples (image_index, image_path))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()

        # Create a dictionary mapping image_index to image_path
        metadata = {index: path for index, path in results}
        return metadata
    