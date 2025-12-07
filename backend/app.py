from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import uvicorn


# Create RAGPipeline instance
rag_pipeline = RAGPipeline()

app = FastAPI(
    title="RAG Chatbot API",
    description="RAG Q&A Chatbot Backend",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str


class AnswerResponse(BaseModel):
    """Response model containing answer and sources."""
    answer: str
    sources: list[str]


@app.get("/")
def read_root():
    """
    Root endpoint with API information.
    
    Returns:
        Dictionary with API status and available endpoints
    """
    return {
        "message": "RAG Chatbot API",
        "status": "running",
        "endpoints": {
            "GET /": "This message",
            "GET /health": "Health check",
            "POST /ask": "Ask a question (requires ingestion first)"
        }
    }

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question based on ingested documents.
    
    Args:
        request: QuestionRequest containing the question
        
    Returns:
        AnswerResponse with answer and source references
        
    Raises:
        HTTPException: If an error occurs during processing
    """
    try:
        print(f"Received question: {request.question}")
        result = rag_pipeline.get_answer(request.question)
        return AnswerResponse(**result)
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 60)
    print("Starting RAG Chatbot Backend")
    print("=" * 60)
    print("Make sure you have run ingestion first!")
    print("Server will run at: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")