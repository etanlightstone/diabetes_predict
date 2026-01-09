#!/bin/bash
# Job Run History - Domino App Launcher
# This script starts the FastAPI application

set -e

# Install dependencies if needed
pip install -q fastapi uvicorn requests

# Start the FastAPI server
# Bind to 0.0.0.0 and port 8888 as required by Domino apps
uvicorn app:app --host 0.0.0.0 --port 8888
