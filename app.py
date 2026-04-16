import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from src.logger import get_logger
from src.pipeline.index_pipeline import IndexPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.utils import get_uploads_path

load_dotenv()
logger = get_logger(__name__)

# App setup 

app = FastAPI(
    title="PaperMind",
    description="Multilingual Document Intelligence System using RAG",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pipeline singletons 

index_pipeline = IndexPipeline()
query_pipeline = QueryPipeline()

# Request models 

class QueryRequest(BaseModel):
    question:  str
    n_results: int = 5

class SelectionQueryRequest(BaseModel):
    selected_text: str
    source_name:   str
    question:      str
    n_results:     int = 5

# Routes 

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    stats = index_pipeline.get_stats()
    return {
        "status":          "healthy",
        "total_documents": stats["total_documents"],
        "total_chunks":    stats["total_chunks"],
    }


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    save_path = get_uploads_path() / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"PDF uploaded: {file.filename}")
    result = index_pipeline.ingest_pdf(str(save_path))

    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result["status"])

    return result


@app.post("/ingest/image")
async def ingest_image(file: UploadFile = File(...)):
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    suffix  = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {allowed}"
        )

    save_path = get_uploads_path() / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Image uploaded: {file.filename}")
    result = index_pipeline.ingest_image(str(save_path))

    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result["status"])

    return result


@app.post("/query")
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = query_pipeline.query(
        question=request.question,
        n_results=request.n_results,
    )
    return result


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    def token_generator():
        for token in query_pipeline.query_stream(
            question=request.question,
            n_results=request.n_results,
        ):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.post("/query/selection")
async def query_selection(request: SelectionQueryRequest):
    if not request.selected_text.strip():
        raise HTTPException(status_code=400, detail="Selected text cannot be empty")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = query_pipeline.query_selection(
        selected_text=request.selected_text,
        source_name=request.source_name,
        question=request.question,
        n_results=request.n_results,
    )
    return result


@app.get("/documents")
async def list_documents():
    docs = index_pipeline.list_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{source_name}")
async def delete_document(source_name: str):
    result = index_pipeline.delete_document(source_name)
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result["status"])
    return result


@app.post("/memory/clear")
async def clear_memory():
    query_pipeline.clear_memory()
    return {"status": "memory cleared"}


@app.get("/evaluate")
async def evaluate():
    try:
        evaluator = RagasEvaluator()
        scores    = evaluator.evaluate()
        return {"status": "success", "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Entry point 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)