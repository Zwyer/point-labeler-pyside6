@echo off
setlocal

cd /d "%~dp0"

set "PYTHONPATH=src;%PYTHONPATH%"

python -m point_labeler.app.main
if errorlevel 1 (
  echo.
  echo App exited with an error.
  exit /b 1
)

exit /b 0

