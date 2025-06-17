from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    fileId: str
    
class QueryResponse(BaseModel):
    answer: str
