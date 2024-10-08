from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_extraction import scrape_wikipedia_page
from milvus_client import MilvusDB

# Initialize FastAPI app
app = FastAPI()

# Initialize MilvusDB
milvus_db = MilvusDB()

# Define request models
class LoadRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

# Store the sentences after loading
sentences = []

@app.post("/load")
async def load_data(request: LoadRequest):
    global sentences
    try:
        content = scrape_wikipedia_page(request.url)
        sentences, _ = milvus_db.load_data(content)
        return {"message": "Data loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_data(request: QueryRequest):
    try:
        results, distances = milvus_db.query(request.query)
        return {"results": results, "distances": distances}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)