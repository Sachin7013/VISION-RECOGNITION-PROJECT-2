from fastapi import APIRouter, UploadFile, File
from app.services import video_processor
from app.utils.embeddings import embedder, embedding_index, embedding_metadata

router = APIRouter()

@router.post("/")
async def upload_video(file: UploadFile = File(...)):
    result = await video_processor.process_video(file)
    return result
