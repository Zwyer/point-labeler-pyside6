@echo off
setlocal

cd /d "%~dp0"

echo [1/3] Checking Python...
python --version >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python not found in PATH.
  exit /b 1
)

echo [2/3] Checking PyInstaller...
python -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
  echo ERROR: PyInstaller is not installed in current environment.
  echo Hint: pip install pyinstaller
  exit /b 1
)

if not exist "PointLabeler.spec" (
  echo ERROR: PointLabeler.spec not found in project root.
  exit /b 1
)

echo [3/3] Building PointLabeler...
python -m PyInstaller --noconfirm --clean "PointLabeler.spec"
if errorlevel 1 (
  echo.
  echo Build failed.
  exit /b 1
)

echo.
echo Build completed successfully.
echo Output: dist\PointLabeler\PointLabeler.exe
exit /b 0

