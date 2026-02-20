"""
K-Space Analysis Web Application
==================================
FastAPI backend for DICOM K-Space analysis.
"""

import os
import sys
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .analysis_engine import run_full_analysis, compare_studies, generate_3d_points, load_dicom_series

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="K-Space Analysis Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
STUDIES_FILE = BASE_DIR / "studies.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Studies storage (JSON file)
# ---------------------------------------------------------------------------

def load_studies():
    if STUDIES_FILE.exists():
        with open(STUDIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_studies(studies):
    with open(STUDIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(studies, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding='utf-8'))


@app.post("/api/upload")
async def upload_dicom(files: list[UploadFile] = File(...)):
    """Upload DICOM files and create a new study."""
    study_id = str(uuid.uuid4())[:8]
    study_dir = UPLOAD_DIR / study_id

    study_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for f in files:
        file_path = study_dir / f.filename
        with open(file_path, "wb") as out:
            content = await f.read()
            out.write(content)
        saved_files.append(f.filename)

    # Create study record
    studies = load_studies()
    studies[study_id] = {
        'id': study_id,
        'created_at': datetime.now().isoformat(),
        'file_count': len(saved_files),
        'status': 'uploaded',
        'metadata': {},
        'results': None,
    }
    save_studies(studies)

    return {
        'study_id': study_id,
        'file_count': len(saved_files),
        'status': 'uploaded',
        'message': f'{len(saved_files)} files uploaded successfully',
    }


@app.get("/api/studies")
async def list_studies():
    """List all studies."""
    studies = load_studies()
    # Return as list sorted by date
    study_list = sorted(studies.values(), key=lambda x: x['created_at'], reverse=True)
    return study_list


@app.get("/api/study/{study_id}")
async def get_study(study_id: str):
    """Get study details and results."""
    studies = load_studies()
    if study_id not in studies:
        raise HTTPException(status_code=404, detail="Study not found")
    return studies[study_id]


@app.post("/api/analyze/{study_id}")
async def analyze_study(study_id: str, background_tasks: BackgroundTasks):
    """Run K-Space analysis on uploaded study."""
    studies = load_studies()
    if study_id not in studies:
        raise HTTPException(status_code=404, detail="Study not found")

    study = studies[study_id]
    if study['status'] == 'analyzing':
        return {'message': 'Analysis already in progress'}

    # Update status
    study['status'] = 'analyzing'
    save_studies(studies)

    # Run analysis in background
    background_tasks.add_task(_run_analysis, study_id)

    return {'message': 'Analysis started', 'study_id': study_id}


def _run_analysis(study_id: str):
    """Background task: run the full analysis pipeline."""
    studies = load_studies()
    study = studies[study_id]

    dicom_path = str(UPLOAD_DIR / study_id)
    output_dir = str(RESULTS_DIR / study_id)

    try:
        results = run_full_analysis(dicom_path, output_dir)

        if 'error' in results:
            study['status'] = 'error'
            study['error'] = results['error']
        else:
            study['status'] = 'completed'
            study['metadata'] = results['metadata']
            study['results'] = {
                'summary': results['summary'],
                'transitions': results['transitions'],
                'radiomics': results['radiomics'],
                'phase': results['phase'],
                'plots': results['plots'],
                'ai_insights': results.get('ai_insights', None)
            }
    except Exception as e:
        study['status'] = 'error'
        study['error'] = str(e)

    studies[study_id] = study
    save_studies(studies)


@app.delete("/api/study/{study_id}")
async def delete_study(study_id: str):
    """Delete a study and its files."""
    studies = load_studies()
    if study_id not in studies:
        raise HTTPException(status_code=404, detail="Study not found")

    # Remove files
    upload_path = UPLOAD_DIR / study_id
    results_path = RESULTS_DIR / study_id
    if upload_path.exists():
        shutil.rmtree(upload_path)
    if results_path.exists():
        shutil.rmtree(results_path)

    del studies[study_id]
    save_studies(studies)

    return {'message': 'Study deleted'}


@app.get("/api/report/{study_id}")
async def get_report(study_id: str):
    """Get analysis report as structured data."""
    studies = load_studies()
    if study_id not in studies:
        raise HTTPException(status_code=404, detail="Study not found")

    study = studies[study_id]
    if study['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Analysis not completed yet")

    return {
        'study_id': study_id,
        'metadata': study['metadata'],
        'results': study['results'],
        'report_time': datetime.now().isoformat(),
    }


@app.get("/api/ai-insights/{study_id}")
async def get_ai_insights(study_id: str):
    """Get Deep K-Space Virtual Biopsy and Pathology Detection results."""
    studies = load_studies()
    if study_id not in studies:
        raise HTTPException(status_code=404, detail="Study not found")

    study = studies[study_id]
    if study['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Analysis not completed yet")

    # ai_insights were added to study['results']['ai_insights']
    insights = study['results'].get('ai_insights', None)
    if not insights:
        raise HTTPException(status_code=404, detail="AI Insights not available for this study (re-analyze needed).")
        
    return {
        'study_id': study_id,
        'insights': insights
    }

# ---------------------------------------------------------------------------
# NEW ENDPOINTS: COMPARISON & 3D
# ---------------------------------------------------------------------------

class ComparisonRequest(BaseModel):
    study_id_1: str
    study_id_2: str

@app.post("/api/compare")
async def compare_api(req: ComparisonRequest):
    """Compare two studies."""
    studies = load_studies()
    if req.study_id_1 not in studies or req.study_id_2 not in studies:
        raise HTTPException(status_code=404, detail="One or both studies not found")
        
    s1 = studies[req.study_id_1]
    s2 = studies[req.study_id_2]
    
    # Load images
    # Note: This is synchronous and might block. For production, offload to worker.
    # But loading from disk is fast enough for this demo.
    path1 = str(UPLOAD_DIR / req.study_id_1)
    path2 = str(UPLOAD_DIR / req.study_id_2)
    
    imgs1, _ = load_dicom_series(path1)
    imgs2, _ = load_dicom_series(path2)
    
    if not imgs1 or not imgs2:
        raise HTTPException(status_code=400, detail="Could not load images")
        
    # Output dir for comparison
    comp_id = f"{req.study_id_1}_vs_{req.study_id_2}"
    out_dir = RESULTS_DIR / "comparisons" / comp_id
    
    try:
        results = compare_studies(req.study_id_1, req.study_id_2, imgs1, imgs2, str(out_dir))
        
        # Return results URL
        return {
            'metrics': results['metrics'],
            'plot_url': f"/static/results/comparisons/{comp_id}/{results['plot']}",
            'slice_count': results['slice_count']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/volume/{study_id}")
async def get_volume_data(study_id: str):
    """Get 3D point cloud data for a study."""
    # Check study
    # Default to getting images from upload dir
    path = str(UPLOAD_DIR / study_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Study files not found")
        
    imgs, _ = load_dicom_series(path)
    if not imgs:
        raise HTTPException(status_code=400, detail="No images found")
        
    try:
        points = generate_3d_points(imgs)
        return {
            'study_id': study_id,
            'points': points,
            'count': len(points)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
