# app.py - FastAPI Backend for CreditChek API Assistant
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.bot_service import BotService
import os
from dotenv import load_dotenv
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="CreditChek API Assistant",
    description="Backend for Mark Musk - CreditChek API Assistant Chatbot",
    version="1.0.0"
)

# CORS configuration - adjust in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - tighten this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize bot service
try:
    bot_service = BotService("")
    logger.info("Bot service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize bot service: {e}")
    raise

# Pydantic models for request validation
class Message(BaseModel):
    content: str
    role: str = "user"

class GitHubRepo(BaseModel):
    url: str
    branch: str = "main"

class DocumentProcessResponse(BaseModel):
    status: str
    message: str
    filename: Optional[str] = None

# Health check endpoint
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "CreditChek API Assistant",
        "version": "1.0.0"
    }

# Chat endpoint
@app.post("/api/chat", response_model=dict)
async def chat(message: Message):
    """
    Process user message and return AI response
    """
    try:
        logger.info(f"Processing chat message: {message.content[:50]}...")
        response = bot_service.generate_response(message.content)
        logger.info("Successfully generated response")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

# GitHub repo processing endpoint
@app.post("/api/process-github", response_model=DocumentProcessResponse)
async def process_github(repo: GitHubRepo):
    """
    Process GitHub repository and index its contents
    """
    try:
        logger.info(f"Processing GitHub repo: {repo.url} (branch: {repo.branch})")
        bot_service.process_github_repo(repo.url, repo.branch)
        logger.info("Successfully processed GitHub repository")
        return {
            "status": "success",
            "message": f"Repository {repo.url} processed successfully"
        }
    except ValueError as ve:
        logger.error(f"Validation error processing GitHub repo: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error processing GitHub repository: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing repository: {str(e)}"
        )

# Document processing endpoint
@app.post("/api/process-document", response_model=DocumentProcessResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Process uploaded document (PDF or TXT) and index its contents
    """
    try:
        logger.info(f"Processing uploaded document: {file.filename}")
        
        # Validate file type
        if file.content_type not in ["application/pdf", "text/plain"]:
            raise ValueError("Unsupported file type. Only PDF and TXT files are accepted.")
        
        # Process the file
        bot_service.process_uploaded_doc(file)
        logger.info(f"Successfully processed document: {file.filename}")
        
        return {
            "status": "success",
            "message": f"Document {file.filename} processed successfully",
            "filename": file.filename
        }
    except ValueError as ve:
        logger.error(f"Validation error processing document: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # Changed from just 'app' to 'app:app'
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )