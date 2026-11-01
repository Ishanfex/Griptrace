@echo off
REM Finishes the 3 sample clips with the final code: re-runs the bench (SKL-007) and carpentry
REM (SGF-021) clips - the gauge clip (mytronlabs-002) is already done - then writes the summary
REM table and the 2-minute demo video. About 30 minutes. Log: samples_log.txt
cd /d "%~dp0"
set LOG=%~dp0samples_log.txt
set PYTHONUNBUFFERED=1
echo GripTrace samples %date% %time% > "%LOG%"
call .venv\Scripts\activate.bat
echo.
echo  [1/3] bench clip MTL-SKL00089-007 (about 15 minutes) - leave this window open
python run.py egohack-data\clips\MTL-SKL00089-007.mp4 >> "%LOG%" 2>&1
echo  [2/3] carpentry clip MTL-SGF00106-021 (about 15 minutes)
python run.py egohack-data\clips\MTL-SGF00106-021.mp4 >> "%LOG%" 2>&1
echo  [3/3] summary table + 2-minute demo video
python -m tools.summary >> "%LOG%" 2>&1
python -m tools.make_demo >> "%LOG%" 2>&1
echo SAMPLES_DONE >> "%LOG%"
echo.
echo  Done. Tell Claude "samples finished".
pause
