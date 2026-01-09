"""
Job Run History Statistics - Domino App.
A FastAPI application that displays job run history as a timeseries bar chart.
"""

import os
import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Job Run History", description="Visualize job run history for this project")

# Configuration from environment
DOMINO_API_HOST = os.environ.get("DOMINO_API_HOST", "")
DOMINO_PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME", "")
DOMINO_PROJECT_OWNER = os.environ.get("DOMINO_PROJECT_OWNER", "")
API_KEY_OVERRIDE = os.environ.get("API_KEY_OVERRIDE", "")


def get_auth_headers():
    """Get authentication headers for Domino API calls."""
    if API_KEY_OVERRIDE:
        return {"X-Domino-Api-Key": API_KEY_OVERRIDE}
    else:
        # Re-acquire token on every call as it expires quickly
        try:
            response = requests.get("http://localhost:8899/access-token")
            token = response.text.strip()
            return {"authorization": token}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get access token: {str(e)}")


def get_project_id():
    """Get the project ID from owner and project name."""
    if not DOMINO_API_HOST or not DOMINO_PROJECT_NAME or not DOMINO_PROJECT_OWNER:
        raise HTTPException(
            status_code=500,
            detail="Missing environment variables: DOMINO_API_HOST, DOMINO_PROJECT_NAME, or DOMINO_PROJECT_OWNER"
        )
    
    headers = get_auth_headers()
    url = f"{DOMINO_API_HOST}/v4/gateway/projects/findProjectByOwnerAndName"
    params = {"ownerName": DOMINO_PROJECT_OWNER, "projectName": DOMINO_PROJECT_NAME}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        project_data = response.json()
        return project_data.get("id")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project ID: {str(e)}")


class JobHistoryItem(BaseModel):
    date: str
    succeeded: int
    failed: int
    stopped: int
    running: int
    queued: int


class JobHistoryResponse(BaseModel):
    jobs: List[dict]
    aggregated: List[JobHistoryItem]
    start_date: str
    end_date: str


def check_environment_config():
    """Check if required environment variables are configured."""
    missing_vars = []
    
    # If running outside Domino (no access token endpoint), need API_KEY_OVERRIDE
    if not API_KEY_OVERRIDE:
        # Check if we can get an access token (running inside Domino)
        try:
            response = requests.get("http://localhost:8899/access-token", timeout=2)
            if response.status_code != 200:
                missing_vars.append("API_KEY_OVERRIDE (required when running outside Domino)")
        except:
            missing_vars.append("API_KEY_OVERRIDE (required when running outside Domino)")
    
    if not DOMINO_API_HOST:
        missing_vars.append("DOMINO_API_HOST")
    if not DOMINO_PROJECT_NAME:
        missing_vars.append("DOMINO_PROJECT_NAME")
    if not DOMINO_PROJECT_OWNER:
        missing_vars.append("DOMINO_PROJECT_OWNER")
    
    return missing_vars


@app.get("/api/job-history")
async def get_job_history(days: int = 30) -> JobHistoryResponse:
    """
    Get job run history for the project.
    Returns aggregated daily counts by status.
    """
    # Check environment configuration - prompt user if running locally without config
    missing_vars = check_environment_config()
    if missing_vars:
        raise HTTPException(
            status_code=503,
            detail=f"App is not fully configured. Please set the following environment variables: {', '.join(missing_vars)}. "
                   f"If running locally outside Domino, set API_KEY_OVERRIDE with your Domino API key."
        )
    
    headers = get_auth_headers()
    project_id = get_project_id()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get jobs from the API
    url = f"{DOMINO_API_HOST}/v4/jobs"
    params = {
        "projectId": project_id,
        "page_size": 1000,  # Get a large batch
        "sort_by": "desc",
        "order_by": "stageTime.submissionTime"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        job_data = response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch jobs: {str(e)}")
    
    jobs = job_data.get("jobs", [])
    
    # Initialize daily buckets for the entire range
    daily_stats = {}
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        daily_stats[date_str] = {
            "date": date_str,
            "succeeded": 0,
            "failed": 0,
            "stopped": 0,
            "running": 0,
            "queued": 0
        }
        current_date += timedelta(days=1)
    
    # Process jobs and aggregate by day
    for job in jobs:
        stage_time = job.get("stageTime", {})
        submission_time = stage_time.get("submissionTime")
        
        if submission_time:
            # Convert milliseconds to datetime
            job_date = datetime.fromtimestamp(submission_time / 1000)
            
            # Only include jobs within the date range
            if start_date <= job_date <= end_date:
                date_str = job_date.strftime("%Y-%m-%d")
                
                if date_str in daily_stats:
                    statuses = job.get("statuses", {})
                    exec_status = statuses.get("executionStatus", "").lower()
                    
                    if exec_status in ["succeeded", "success", "completed"]:
                        daily_stats[date_str]["succeeded"] += 1
                    elif exec_status in ["failed", "error"]:
                        daily_stats[date_str]["failed"] += 1
                    elif exec_status in ["stopped", "cancelled"]:
                        daily_stats[date_str]["stopped"] += 1
                    elif exec_status in ["running", "executing"]:
                        daily_stats[date_str]["running"] += 1
                    elif exec_status in ["queued", "pending"]:
                        daily_stats[date_str]["queued"] += 1
                    else:
                        # Count unknown statuses as queued
                        daily_stats[date_str]["queued"] += 1
    
    # Convert to sorted list
    aggregated = sorted(daily_stats.values(), key=lambda x: x["date"])
    
    return JobHistoryResponse(
        jobs=jobs,
        aggregated=aggregated,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
