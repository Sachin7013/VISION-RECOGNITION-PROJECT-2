from fastapi import FastAPI
from app.routers import upload, query

app = FastAPI()

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(query.router, prefix="/query", tags=["query"])

@app.get("/")
def home():
    return {"message": "Welcome to the API"}
