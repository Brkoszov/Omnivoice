@echo off
setlocal
cd /d "%~dp0"

echo Starting OmniVoice...
echo Using isolated environment in .venv...

:: Set PATH to include uv and the virtual environment
set "PATH=C:\Users\ayaka\.local\bin;%CD%\.venv\Scripts;%CD%\.venv\bin;%PATH%"

:: Run the application using uv run (which handles the venv automatically)
uv run app.py

pause