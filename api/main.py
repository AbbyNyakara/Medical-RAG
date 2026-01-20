'''
Docstring for api.main

Creates a minimalistic API to access the medical RAG
'''
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.rag.pipeline import MedicalRAGPipeline
from typing import Optional
from contextlib import asynccontextmanager
import logging
import tempfile #generates tempoorary files and directories
import os
import uvicorn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# AWS Textract can support many formats - SUPPORTED_FORMATS = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.webp', '.bmp'}
# Global pipeline instance - for Medical RAG Pipeline to be reused
pipeline: Optional[MedicalRAGPipeline] = None


@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Application lifespan manager
    - Startup: Initialize pipeline before requests
    - Shutdown: Cleanup resources after requests
    """
    global pipeline

    logger.info("Starting RAG API")

    try:
        pipeline = MedicalRAGPipeline(
            s3_bucket="medical-rag-docs-abigael-2026",
            llm_config={
                'model': 'gpt-4-turbo',
                'temperature': 0.2,
                'max_tokens': 500
            })
        logger.info("Pipeline initialized")
    except Exception as e:
        logger.error("Failed to initialize pipeline %s", e)

    yield #Yield control after setup and code below yield will run on shutdown

    logger.info("Shutting down API")
    # Cleanup if needed - delete the Embeddings, the docs form s3 bucket
    pipeline = None
    logger.info("Cleanup complete")

app = FastAPI(
    title="Medical RAG API",
    description="Upload medical documents and ask questions",
    version='1.0.0',
    lifespan=lifespan
)


## Request and Response models

class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    success: bool
    document_id: Optional[str] = None
    filename: str
    total_chunks: int
    vectors_stored: int
    message: str
    error: Optional[str] = None

class QueryRequest(BaseModel):
    """Request for asking a question"""
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response for a question"""
    success: bool
    question: str
    answer: str
    num_sources: int
    processing_time_seconds: float
    error: Optional[str] = None

# class PipelineStats(BaseModel):
#     """Pipeline statistics"""
#     total_vectors: int
#     dimension: int
#     is_ready: bool

## Create the endpoints:
@app.post('/upload', response_model=DocumentUploadResponse, summary="Upload and index document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF file for indexing
    - Extracts text, chunks content, generates embeddings
    - Stores vectors in Pinecone for querying
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF Files allowed") # Check which files allow for aws texttract

   
    try:
        # Todo 1 - Save file to temp locatin
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        logger.info("Uploaded file to %s", tmp_path)
        # Todo 2 - Index document: Extract -> Chunk -> Embed
        result = pipeline.index_document(file_path=tmp_path) 
        os.unlink(tmp_path)

         # Step 4: Return response
        if result['success']:
            return DocumentUploadResponse(
                success=True,
                document_id=result.get('document_id'),
                filename=file.filename,
                total_chunks=result.get('total_chunks', 0),
                vectors_stored=result.get('embedding', {}).get('vectors_stored', 0),
                message=f"Document indexed: {result.get('total_chunks')} chunks stored in vector DB"
            )
        else:
            return DocumentUploadResponse(
                success=False,
                filename=file.filename,
                total_chunks=0,
                vectors_stored=0,
                message="Indexing failed",
                error=result.get('error', 'Unknown error')
            )
    except Exception as e:
        logger.error("Error indexing document %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )
    

@app.post('/query', response_model=QueryResponse, summary="Ask question from uploaded document")
async def ask_question(request: QueryRequest):
    """
    Allows user to ask question on the indexed document. 
    """
    if not request.question.strip():
        raise HTTPException(500, detail="Question cannot be empty")
    try:
        logger.info("Processing Query %s", request.question[:100])

        # Get answer from the pipeline:
        result = pipeline.answer_question(query=request.question, top_k=request.top_k)
        if result['success']:
            return QueryResponse(
                success=True,
                question=request.question,
                answer=result['answer'],
                num_sources=result['num_sources'],
                processing_time_seconds=result.get('processing_time_seconds', 0)
            )
        else:
            return QueryResponse(
                success=False,
                question=request.question,
                answer="",
                num_sources=0,
                processing_time_seconds=0,
                error=result.get('error', 'Unknown error')
            )
        
    except Exception as e:
        logger.error("Query Failed %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer {e}")
    

app.get('/heath')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None
    }

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Medical RAG API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload (POST) - Upload PDF documents",
            "query": "/query (POST) - Ask questions",
            "health": "/health (GET) - Health check",
            "docs": "/docs - Interactive API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

