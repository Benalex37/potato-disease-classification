@echo off
REM Starts the FastAPI prediction server on http://localhost:8000
cd /d "%~dp0api"
"%~dp0.venv\Scripts\python.exe" main.py
