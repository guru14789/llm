#!/usr/bin/env python3
"""
AI Document Query System - Professional Intelligence Platform
Advanced document analysis with AI-powered insights
"""
import os
import logging
import time
import requests
import re
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai

# Try PyMuPDF first, fallback to pdfminer
try:
    import fitz  # PyMuPDF
    PDF_LIBRARY = "pymupdf"
except ImportError:
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        import io
        PDF_LIBRARY = "pdfminer"
    except ImportError:
        PDF_LIBRARY = "none"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini AI configured successfully")
else:
    logger.error("GEMINI_API_KEY not found in environment variables")

# FastAPI app
app = FastAPI(
    title="AI Document Intelligence Platform",
    description="Advanced AI-powered document analysis and query system for professional use",
    version="2.0.0"
)

# Templates configuration
templates = Jinja2Templates(directory="templates")

# Security
security = HTTPBearer()
BEARER_TOKEN = "12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Data models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Simple document cache
document_cache = {}

def get_gemini_model():
    """Get Gemini model - try available models"""
    try:
        # Try the latest available models
        for model_name in ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"Successfully loaded model: {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue

        logger.error("No Gemini models available")
        return None
    except Exception as e:
        logger.error(f"Error getting Gemini model: {e}")
        return None

async def download_document(url: str) -> bytes:
    """Download document from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise Exception(f"Failed to download document: {e}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF using best available method"""
    try:
        if PDF_LIBRARY == "pymupdf":
            # Use PyMuPDF (faster and more reliable)
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            logger.info(f"Extracted {len(text)} characters using PyMuPDF")
            return text

        elif PDF_LIBRARY == "pdfminer":
            # Fallback to pdfminer
            laparams = LAParams(
                boxes_flow=0.5,
                word_margin=0.1,
                char_margin=2.0,
                line_margin=0.5
            )
            text = extract_text(io.BytesIO(pdf_content), laparams=laparams)
            logger.info(f"Extracted {len(text)} characters using pdfminer")
            return text
        else:
            raise Exception("No PDF extraction library available")

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        raise Exception(f"Failed to extract PDF text: {e}")

def clean_text(text: str) -> str:
    """Clean extracted text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """Split text into optimized chunks for better accuracy"""
    chunks = []

    # Split by sentences first for better context preservation
    sentences = re.split(r'[.!?]+', text)

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence)

        # If adding this sentence would exceed chunk size and we have content
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 2  # +2 for '. '

    # Add the last chunk
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    # Filter out very short chunks (less than 50 characters)
    chunks = [chunk for chunk in chunks if len(chunk) > 50]

    return chunks

def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 4) -> List[str]:
    """Find relevant chunks using enhanced keyword matching and scoring"""
    question_lower = question.lower()
    question_words = set(question_lower.split())

    # Enhanced keyword sets for better matching
    insurance_keywords = {
        'policy', 'coverage', 'benefit', 'premium', 'insured', 'claim', 'deductible',
        'copayment', 'waiting', 'grace', 'exclusion', 'maternity', 'ayush', 'room rent',
        'sum insured', 'pre-existing', 'cashless', 'reimbursement', 'hospitalization'
    }

    medical_keywords = {
        'treatment', 'hospital', 'medical', 'doctor', 'surgery', 'diagnosis',
        'illness', 'disease', 'condition', 'therapy', 'medication', 'consultation'
    }

    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split())

        # 1. Basic word overlap score
        overlap = len(question_words.intersection(chunk_words))

        # 2. Insurance domain relevance
        insurance_matches = sum(1 for keyword in insurance_keywords if keyword in chunk_lower)

        # 3. Medical domain relevance
        medical_matches = sum(1 for keyword in medical_keywords if keyword in chunk_lower)

        # 4. Question-specific keyword bonuses
        specific_bonus = 0
        if 'waiting' in question_lower and 'waiting' in chunk_lower:
            specific_bonus += 2
        if 'grace' in question_lower and 'grace' in chunk_lower:
            specific_bonus += 2
        if 'ayush' in question_lower and 'ayush' in chunk_lower:
            specific_bonus += 2
        if 'exclusion' in question_lower and any(term in chunk_lower for term in ['exclusion', 'excluded', 'not covered']):
            specific_bonus += 2
        if 'maternity' in question_lower and 'maternity' in chunk_lower:
            specific_bonus += 2
        if 'company' in question_lower and any(term in chunk_lower for term in ['hdfc', 'ergo', 'company', 'limited']):
            specific_bonus += 2

        # 5. Length quality factor (prefer substantial chunks)
        length_bonus = min(1.0, len(chunk) / 200)

        # Calculate final score
        total_score = (
            overlap * 1.0 +                    # Word overlap
            insurance_matches * 0.8 +          # Insurance relevance
            medical_matches * 0.3 +            # Medical relevance
            specific_bonus * 1.5 +             # Question-specific matches
            length_bonus * 0.5                 # Length quality
        )

        if total_score > 0:
            scored_chunks.append((chunk, total_score))

    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]

async def answer_question(question: str, relevant_chunks: List[str]) -> str:
    """Generate answer using Gemini"""
    try:
        model = get_gemini_model()
        if model is None:
            return "Error: AI service is not available"

        if not relevant_chunks:
            return "This information is not available in the provided document"
        
        # Prepare context
        context = "\n\n".join([f"Section {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
        
        # Enhanced prompt for maximum accuracy
        prompt = f"""You are an expert insurance policy analyst. Analyze the following document sections and answer the question with precision.

DOCUMENT SECTIONS:
{context}

QUESTION: {question}

ANALYSIS INSTRUCTIONS:
1. CAREFULLY read each document section above
2. Look for EXACT information related to the question
3. Include SPECIFIC details: numbers, percentages, time periods, amounts, conditions
4. Quote or reference relevant policy text when providing specific information
5. If information is not explicitly stated, respond: "This information is not available in the provided document"

RESPONSE GUIDELINES:
- For company questions: Look for company names, registration numbers, addresses
- For waiting periods: Look for specific time periods (days, months, years)
- For grace periods: Look for payment deadlines and grace allowances
- For coverage: Look for benefit amounts, limits, conditions
- For exclusions: Look for "excluded", "not covered", "limitations"
- For AYUSH: Look for alternative medicine, AYUSH hospitals, traditional treatments
- For maternity: Look for pregnancy, childbirth, maternity benefits

FORMAT YOUR ANSWER:
- Start with a direct answer
- Include specific details from the document
- Reference policy sections when relevant
- Be concise but comprehensive

ANSWER:"""

        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Error: No response generated"
            
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error: Unable to generate answer - {str(e)}"

async def process_document_and_questions(document_url: str, questions: List[str]) -> List[str]:
    """Process document and answer questions"""
    try:
        # Check cache
        if document_url in document_cache:
            chunks = document_cache[document_url]
        else:
            # Download and process document
            content = await download_document(document_url)
            text = extract_text_from_pdf(content)
            cleaned_text = clean_text(text)
            chunks = chunk_text(cleaned_text)
            document_cache[document_url] = chunks
        
        # Answer questions
        answers = []
        for question in questions:
            relevant_chunks = find_relevant_chunks(question, chunks)
            answer = await answer_question(question, relevant_chunks)
            answers.append(answer)
        
        return answers
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return [f"Error: Document processing failed - {str(e)}"] * len(questions)

@app.get("/ping")
async def ping():
    """Simple ping endpoint for testing"""
    return {"status": "ok", "message": "pong"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Ultra-minimal LLM Query Retrieval System - Phase 1",
        "timestamp": time.time(),
        "version": "1.0.0",
        "pdf_library": PDF_LIBRARY,
        "gemini_configured": GEMINI_API_KEY is not None
    }

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint serving the original frontend template"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def hackrx_document_processing(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing document queries"""
    try:
        start_time = time.time()
        
        # Process document and questions
        answers = await process_document_and_questions(request.documents, request.questions)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(request.questions)} questions in {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in hackrx_document_processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    
    try:
        port = int(os.environ.get("PORT", 8000))
    except (TypeError, ValueError):
        port = 8000
    
    uvicorn.run(app, host="0.0.0.0", port=port)