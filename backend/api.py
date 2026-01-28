"""
FastAPI Backend for Stutter Analysis
====================================
REST API that serves the stutter analysis system.

Endpoints:
- POST /analyze - Upload audio and get analysis
- GET /report/{id} - Get a saved report
- GET /health - Health check

This connects the React frontend to our Python ML models.
"""

import os
import sys
import shutil
from datetime import datetime
from typing import Optional
import json

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import our modules
from inference.clinical_report import generate_clinical_report, save_report_json
from inference.WaveLM_inference import load_wavlm_model
from inference.Whisper_inference import load_whisper_model

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Stutter Analysis API",
    description="Multimodal stuttering analysis using WavLM and Whisper",
    version="1.0.0"
)

# Enable CORS (allows React frontend to call this API)
# In production, replace "*" with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server runs on localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL MODEL LOADING (Load once at startup for faster inference)
# ============================================================================

# These will be loaded when the server starts
wavlm_model = None
wavlm_device = None
whisper_model = None

@app.on_event("startup")
async def load_models():
    """Load ML models when server starts (only once)."""
    global wavlm_model, wavlm_device, whisper_model
    
    print("=" * 60)
    print("LOADING MODELS AT STARTUP...")
    print("=" * 60)
    
    # Load WavLM model
    print("\nLoading WavLM model...")
    wavlm_model, wavlm_device = load_wavlm_model()
    
    # Load Whisper model
    print("\nLoading Whisper model...")
    whisper_model = load_whisper_model("base")
    
    print("\n" + "=" * 60)
    print("ALL MODELS LOADED - SERVER READY!")
    print("=" * 60)


# ============================================================================
# DIRECTORIES SETUP
# ============================================================================

# Directory to store uploaded audio files temporarily
UPLOAD_DIR = os.path.join(project_root, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory to store reports
REPORTS_DIR = os.path.join(project_root, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "Stutter Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /analyze - Upload audio for analysis",
            "report": "GET /report/{report_id} - Get saved report",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "wavlm": wavlm_model is not None,
            "whisper": whisper_model is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    patient_name: str = Form(default="Anonymous"),
    patient_id: Optional[str] = Form(default=None),
    clinician_name: Optional[str] = Form(default=None),
    threshold: float = Form(default=0.4)
):
    """
    Main endpoint: Upload audio and get stutter analysis.
    
    Parameters:
    - file: Audio file (WAV, MP3, etc.)
    - patient_name: Name for the report
    - patient_id: Optional patient ID
    - clinician_name: Optional clinician name
    - threshold: Detection sensitivity (default 0.4)
    
    Returns:
    - Complete clinical report as JSON
    """
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file temporarily
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"upload_{timestamp}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\nReceived file: {file.filename}")
        print(f"Saved to: {file_path}")
        
        # Generate clinical report using our models
        report = generate_clinical_report(
            audio_path=file_path,
            patient_name=patient_name,
            patient_id=patient_id,
            clinician_name=clinician_name,
            wavlm_model=wavlm_model,
            wavlm_device=wavlm_device,
            whisper_model=whisper_model,
            threshold=threshold
        )
        
        # Save the report
        report_path = save_report_json(report)
        report['report_path'] = report_path
        
        return JSONResponse(content=report)
    
    except Exception as e:
        # Log the error with full traceback
        import traceback
        error_msg = str(e)
        full_traceback = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"ERROR PROCESSING FILE!")
        print(f"{'='*60}")
        print(f"Error: {error_msg}")
        print(f"\nFull Traceback:")
        print(full_traceback)
        print(f"{'='*60}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {error_msg}"
        )
    
    finally:
        # Clean up uploaded file (optional - you might want to keep it)
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/report/{report_id}")
async def get_report(report_id: str):
    """
    Get a previously saved report by ID.
    
    Parameters:
    - report_id: The report ID (e.g., "RPT-20260128-143022")
    
    Returns:
    - The saved report as JSON
    """
    report_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {report_id}"
        )
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    return JSONResponse(content=report)


@app.get("/reports")
async def list_reports():
    """List all saved reports."""
    reports = []
    
    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith('.json'):
            report_id = filename.replace('.json', '')
            file_path = os.path.join(REPORTS_DIR, filename)
            
            # Get basic info from the report
            with open(file_path, 'r') as f:
                report = json.load(f)
            
            reports.append({
                'report_id': report_id,
                'patient_name': report.get('patient_info', {}).get('name', 'Unknown'),
                'generated_at': report.get('generated_at'),
                'severity': report.get('severity', {}).get('label', 'Unknown')
            })
    
    return {"reports": reports, "count": len(reports)}


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    # Run with: python backend/api.py
    # Or: uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
    
    print("\n" + "=" * 60)
    print("STARTING STUTTER ANALYSIS API SERVER")
    print("=" * 60)
    print("\nServer will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (dev mode)
    )
