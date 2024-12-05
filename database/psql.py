import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from util.util import dataset_storage_name


class PSQLDatabase:
    def __init__(self):
        # Load environment variables from the .env file
        load_dotenv()

        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = os.getenv("DB_PORT", "5432")
        self.DB_NAME = os.getenv("DB_NAME", "image_search")
        self.DB_USER = os.getenv("DB_USER")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD")
        self.dataset_table = "datasets"

        self._create_database_if_not_exists()
        self._create_datasets_table_if_not_exists()

    def _create_database_if_not_exists(self):
        # Connect to the default database 'postgres' to check for the target database
        conn = psycopg2.connect(
            dbname="postgres",
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
        )
        conn.set_isolation_level(
            ISOLATION_LEVEL_AUTOCOMMIT
        )  # Allows creating a new database

        cursor = conn.cursor()

        # Check if the database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.DB_NAME}'")
        exists = cursor.fetchone()

        if not exists:
            # If the database does not exist, create it
            cursor.execute(f"CREATE DATABASE {self.DB_NAME}")
            print(f"Database '{self.DB_NAME}' created successfully.")

        cursor.close()
        conn.close()

    def _create_datasets_table_if_not_exists(self):
        conn = psycopg2.connect(
            dbname=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
        )
        cursor = conn.cursor()

        # Create the table if it does not exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.dataset_table} (
            name TEXT PRIMARY KEY,
            size INTEGER
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()

    def _create_metadata_table(self, dataset_name):
        # Create a connection to the PostgreSQL database
        conn = psycopg2.connect(
            host=self.DB_HOST,
            port=self.DB_PORT,
            dbname=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
        )
        cursor = conn.cursor()

        # Dynamically create a table for storing image metadata for a specific dataset
        table_name = dataset_storage_name(
            dataset_name
        )  # Use dataset-specific table name
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                image_index INTEGER,
                image_path TEXT
            )
        """
        )

        conn.commit()
        cursor.close()
        conn.close()

    def _insert_images(self, dataset_name, dataset_size, image_paths):
        # Insert image metadata into a specific dataset's table
        conn = psycopg2.connect(
            host=self.DB_HOST,
            port=self.DB_PORT,
            dbname=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
        )
        cursor = conn.cursor()

        table_name = dataset_storage_name(dataset_name)  # Dataset-specific table
        for idx, path in enumerate(image_paths):
            cursor.execute(
                f"""
                INSERT INTO {table_name} (image_index, image_path)
                VALUES (%s, %s)
            """,
                (idx, path),
            )

        cursor.execute(
            f"""
            INSERT INTO {self.dataset_table} (name, size)
            VALUES (%s, %s)
        """,
            (dataset_name, dataset_size),
        )

        conn.commit()
        cursor.close()
        conn.close()

    def _fetch_images(self, dataset_name, indices):
        # Fetch image paths from the dataset-specific table based on indices
        conn = psycopg2.connect(
            host=self.DB_HOST,
            port=self.DB_PORT,
            dbname=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
        )
        cursor = conn.cursor()

        table_name = dataset_storage_name(dataset_name)  # Dataset-specific table
        # Create a string of placeholders for the number of indices
        placeholders = ", ".join(["%s"] * len(indices))

        # Construct the SQL query with the placeholders
        query = f"""
            SELECT image_index, image_path 
            FROM {table_name}
            WHERE image_index IN ({placeholders})
        """

        # Execute the query with the list of indices
        cursor.execute(query, indices)  # Pass the list directly

        # Fetch the results (returns list of tuples (image_index, image_path))
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return results

    def store_image_paths(self, dataset_name, dataset_size, image_paths):
        self._create_metadata_table(dataset_name)
        self._insert_images(dataset_name, dataset_size, image_paths)

    def fetch_image_paths(self, dataset_name, indices):
        results = self._fetch_images(dataset_name, indices)

        # Create a dictionary mapping image_index to image_path
        image_paths = {index: path for index, path in results}
        return image_paths

    def get_all_datasets(self):
        # List to store dataset names
        available_datasets = []

        # Connect to the database
        conn = psycopg2.connect(
            host=self.DB_HOST,
            port=self.DB_PORT,
            dbname=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
        )
        cursor = conn.cursor()

        # Query to get all dataset names from the dataset_table
        cursor.execute(f"SELECT name, size FROM {self.dataset_table}")

        # Fetch all dataset names and add them to the available_datasets list
        available_datasets = {row[0]: row[1] for row in cursor.fetchall()}

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return available_datasets

    def get_dataset_size(self, dataset_name):
        return self.get_all_datasets()[dataset_name]
