@echo off
REM One-click setup for Windows: double-click this file.
REM Installs everything into .venv, downloads models, runs tests, finds the clips,
REM and runs the real models on 30 s of one clip. Everything is logged to setup_log.txt
cd /d "%~dp0"
set LOG=%~dp0setup_log.txt
echo GripTrace setup %date% %time% > "%LOG%"
echo.
echo  GripTrace setup - this takes 5-15 minutes. Leave this window open.
echo  Progress is written to setup_log.txt
echo.

REM ---- find Python 3.10-3.12 ----
set PY=
if exist "%USERPROFILE%\Python311\python.exe" set PY="%USERPROFILE%\Python311\python.exe"
if not defined PY (where py >nul 2>&1 && set PY=py -3.11)
if not defined PY set PY=python
echo PY=%PY% >> "%LOG%"
%PY% --version >> "%LOG%" 2>&1

REM ---- git is needed once by YOLO-World to fetch its text model ----
if exist "%USERPROFILE%\MinGit\cmd\git.exe" set PATH=%USERPROFILE%\MinGit\cmd;%PATH%
where git >> "%LOG%" 2>&1

echo [1/5] creating virtual environment
if not exist .venv\Scripts\python.exe %PY% -m venv .venv >> "%LOG%" 2>&1
call .venv\Scripts\activate.bat
echo [2/5] installing packages (mediapipe, opencv, ultralytics...)
python -m pip install --upgrade pip >> "%LOG%" 2>&1
pip install -r requirements.txt >> "%LOG%" 2>&1
echo [3/5] downloading models
python prep.py >> "%LOG%" 2>&1
echo [4/5] running tests
python -m pytest -q >> "%LOG%" 2>&1
echo [5/5] finding clips and running the real models on 30 s
python -m tools.get_clips >> "%LOG%" 2>&1
python -m tools.find_clips >> "%LOG%" 2>&1
python -m tools.smoke_real >> "%LOG%" 2>&1
echo ALL_DONE >> "%LOG%"
echo.
echo  Done. Tell Claude "setup finished".
pause
