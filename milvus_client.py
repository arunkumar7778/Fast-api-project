from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import SentenceTransformer
import numpy as np

# Milvus connection setup
connections.connect(host='localhost', port='19530')

# Define the schema for Milvus collection
def create_milvus_collection(collection_name: str):
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields)
    collection = Collection(name=collection_name, schema=schema)
    
    # Create the index for fast similarity search
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection

class MilvusDB:
    def _init_(self, collection_name="wiki_collection"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = create_milvus_collection(collection_name)
        self.collection.load()

    def load_data(self, data: str):
        sentences = data.split("\n")  # Split into sentences or paragraphs
        embeddings = self.model.encode(sentences)
        
        # Prepare data to insert into Milvus
        entities = [sentences, embeddings.tolist()]
        self.collection.insert(entities)

        # Flush to make sure the data is saved
        self.collection.flush()

        return sentences, embeddings

    def query(self, query: str, top_k=3):
        query_embedding = self.model.encode([query]).tolist()
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=None,
        )
        
        # Extract results and distances
        result_texts = []
        result_distances = []
        for result in results[0]:
            result_texts.append(result.entity.get("text"))
            result_distances.append(result.distance)
        
        return result_texts, result_distances