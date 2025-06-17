from fastapi import FastAPI, UploadFile, File
from models import QueryRequest, QueryResponse
import os
import uuid
import fitz
from sentence_transformers import SentenceTransformer
import chromadb
import time
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "*"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

os.makedirs("data", exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="pdfs")

def read_text(file_path):
    doc = fitz.open(file_path)
    return "".join(page.get_text().lower() for page in doc)

def chunk_text(text, chunk_size=300, overlap=50):
    """Smaller chunks for better performance"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size].strip()
        if len(chunk) > 50:  
            chunks.append(chunk)
    return chunks

def store_chunks(chunks, fileid):
    """Process in smaller batches to avoid memory issues"""
    if not chunks:
        return
    
    batch_size = 50  # Smaller batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        embeddings = model.encode(batch_chunks, batch_size=16)
        
        ids = [f"{fileid}-{j}" for j in range(i, i+len(batch_chunks))]
        metadatas = [{"fileid": fileid} for _ in range(len(batch_chunks))]
        
        collection.add(
            documents=batch_chunks,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF vector search API."}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()
    
    fileid = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1]
    path = os.path.join("data", f"{fileid}{ext}")
    
    with open(path, "wb") as f:
        f.write(await file.read())
    
    text = read_text(path)
    chunks = chunk_text(text)
    store_chunks(chunks, fileid)
    
    processing_time = time.time() - start_time
    
    return {
        "message": "File uploaded and stored successfully.",
        "fileid": fileid,
        "chunks": len(chunks),
        "extension": ext,
        "processing_time": f"{processing_time:.2f}s"
    }

@app.post("/query")
def query_pdf(req: QueryRequest):
    start_time = time.time()
    query_embedding = model.encode([req.query.lower()], batch_size=1)
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,  
        where={"fileid": req.fileId} if req.fileId else None,
        include=["documents", "distances"]
    )
    
    query_time = time.time() - start_time
    
    if results["documents"]:
        context = ''.join(results["documents"][0])
    else:
        context = "No relevant context found."
        
    prompt = f'''
        You will answer this query: {req.query} based on this context: {context}. If context is not provided, simply say the result was not found in the document.
    '''
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    
    llm_response = response.json().get("response", "").strip()

    return {
        "answer": llm_response,
        "results": results["documents"][0] if results["documents"] else [],
        "query_time": f"{query_time:.3f}s",
        "distances": results["distances"][0] if results["distances"] else []
    }
    
@app.get("/debug/collection-size")
def get_collection_size():
    """Debug endpoint to check collection size"""
    try:
        return {"count": collection.count()}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/debug/clear-collection")
def clear_collection():
    """Debug endpoint to clear collection if needed"""
    try:
        all_docs = collection.get()
        if all_docs["ids"]:
            collection.delete(ids=all_docs["ids"])
        return {"message": "Collection cleared"}
    except Exception as e:
        return {"error": str(e)}