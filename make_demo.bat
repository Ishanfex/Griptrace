@echo off
REM Builds outputs\demo_2min.mp4 (under 2 minutes) from the rendered clips, for the private Drive folder.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m tools.make_demo
echo.
echo  Done: outputs\demo_2min.mp4  -  upload it ONLY to the organisers' private Drive folder.
pause
