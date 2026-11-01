@echo off
REM GripTrace app: the viewer + the analysis pipeline in your browser.
REM Double-click this file. Your browser opens at http://127.0.0.1:8765  (only this computer can open it).
REM Keep this window open while you use the app; close it to stop.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo  Setup has not been run yet. Double-click setup_and_test.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python app.py
pause
