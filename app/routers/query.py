from fastapi import APIRouter, Body
from app.services import query_engine
from app.utils.embeddings import embedder, embedding_index, embedding_metadata

router = APIRouter()

@router.post("/")
async def query_llm(question: str = Body(...)):
    result = await query_engine.query_ollama(question)
    return result
