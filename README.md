# Wikipedia Data Extraction and Querying with Milvus and FastAPI

This project is a FastAPI-based web service for extracting data from Wikipedia pages, storing it in Milvus (a vector database), and performing similarity-based searches using pre-trained embeddings from sentence-transformers.

## Features

- *Data Extraction*: Scrape content from Wikipedia pages.
- *Embeddings*: Use Sentence Transformers to embed the text into vector form.
- *Vector Search*: Store and search vectorized data in Milvus, enabling similarity-based retrieval.
- *FastAPI*: Provides two POST endpoints for loading data and querying the Milvus vector database.

Extracts main content from a specified Wikipedia page.
Cleans the extracted text by removing numbers and unnecessary formatting.
Embedding Module: This module provides functionality for generating text embeddings using state-of-the-art models, enabling efficient semantic search and natural language processing tasks. Features:

Generates embeddings for text using the Sentence Transformer model.
Easy integration with various data sources, such as Wikipedia content.
Milvus Embedding Module: This module provides functionality to interact with Milvus, a high-performance vector database, for storing and querying embeddings. It includes functions for creating a collection and inserting embeddings along with their corresponding text. Features:

Create a collection in Milvus for storing embeddings and associated text.
Insert embeddings into the collection.
Automatically manage existing collections by dropping them if they already exist.
Create and load an index on the embedding field for efficient searching.
FastAPI Wikipedia Query Service: This FastAPI application allows users to scrape Wikipedia pages, embed their content, and perform question answering based on the embedded data. It combines various NLP technologies to provide an efficient and effective query service. Features

Load Wikipedia Data: Scrapes content from a specified Wikipedia page and stores it as embeddings in a Milvus vector database.
Query Data: Users can ask questions based on the loaded content, and the system provides answers using a T5 model.
