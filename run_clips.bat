@echo off
REM Process every clip in egohack-data\clips with the full pipeline (hands + objects + contacts).
REM Double-click this file. Results go to outputs\<clip>\ ; the log goes to run_log.txt
cd /d "%~dp0"
set LOG=%~dp0run_log.txt
set PYTHONUNBUFFERED=1
echo GripTrace run %date% %time% > "%LOG%"
call .venv\Scripts\activate.bat
echo.
echo  [1/2] updating the object model for the new object list (about 1 minute)
python prep.py >> "%LOG%" 2>&1
echo  [2/2] processing clips - about 8 minutes per 3-minute clip. Leave this window open.
for %%f in (egohack-data\clips\*.mp4) do (
  echo        %%~nxf
  python run.py "%%f" >> "%LOG%" 2>&1
)
echo ALL_DONE >> "%LOG%"
echo.
echo  Done. Tell Claude "clips finished".
pause
