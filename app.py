from fastapi import FastAPI
from pydantic import BaseModel
from selfQeryRAG import RAGImplementation

app = FastAPI()
rag_implementation = RAGImplementation()

class Query(BaseModel):
    question: str

@app.get("/test")
async def test_question():
    return {"message": "Hello"}

@app.post("/completion")
async def ask_question(query: Query):
    response = rag_implementation.ask(query.question)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)