@echo off
REM Processes EVERY clip in egohack-data\clips that has no results yet (safe to re-run).
REM Demo videos are rendered only for the showcase clips listed below, to save time.
cd /d "%~dp0"
set LOG=%~dp0run_all_log.txt
set PYTHONUNBUFFERED=1
echo GripTrace run_all %date% %time% >> "%LOG%"
call .venv\Scripts\activate.bat
python -m tools.get_clips >> "%LOG%" 2>&1
echo.
echo  Processing all clips (about 8 minutes each). Leave this window open - safe to stop and re-run later.
python run.py egohack-data\clips --all --skip-done --render-only MTL-SKL00089-007,mytronlabs-000003-002,MTL-SGF00106-021 >> "%LOG%" 2>&1
python -m tools.summary >> "%LOG%" 2>&1
echo ALL_DONE >> "%LOG%"
echo.
echo  Done. Tell Claude "all clips finished".
pause
