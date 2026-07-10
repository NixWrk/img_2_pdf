@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "NEEDS_INSTALL=0"

if not exist "%VENV_PY%" (
  echo [UniScan] Creating virtual environment...
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .venv
  ) else (
    py -3.11 -m venv .venv
    if errorlevel 1 py -3 -m venv .venv
  )
  if errorlevel 1 python -m venv .venv
  if errorlevel 1 goto :error
  "%VENV_PY%" -c "import sys; raise SystemExit(sys.version_info < (3, 11))"
  if errorlevel 1 (
    echo [UniScan] Python 3.11 or newer is required.
    goto :error
  )
  set "NEEDS_INSTALL=1"
)

if "%NEEDS_INSTALL%"=="0" (
  "%VENV_PY%" -c "import cv2, customtkinter, fitz, img2pdf, onnxruntime, uniscan" >nul 2>nul
  if errorlevel 1 set "NEEDS_INSTALL=1"
)

if "%NEEDS_INSTALL%"=="1" (
  echo [UniScan] Installing dependencies...
  "%VENV_PY%" -m pip install -e .
  if errorlevel 1 goto :error
)

echo [UniScan] Launching application...
"%VENV_PY%" -m uniscan.cli %*
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
  echo [UniScan] Application exited with code %APP_EXIT%.
)

exit /b %APP_EXIT%

:error
echo [UniScan] Startup failed.
exit /b 1
