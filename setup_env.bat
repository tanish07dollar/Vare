@echo off
echo ============================================
echo  VARE — Environment Setup
echo ============================================

:: Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv vare_env
if errorlevel 1 (echo ERROR: python not found. Install Python 3.11 from python.org & pause & exit /b 1)

:: Activate it
call vare_env\Scripts\activate.bat

:: Install PyTorch CPU-only first (lighter, no CUDA needed)
echo [2/4] Installing PyTorch (CPU)...
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

:: Install remaining packages
echo [3/4] Installing remaining packages...
pip install transformers>=4.41.0 accelerate safetensors huggingface-hub
pip install soundfile scipy fastapi "uvicorn[standard]" tqdm PyYAML numpy==1.26.4

echo [4/4] Done!
echo.
echo To start the server:
echo   vare_env\Scripts\activate.bat
echo   python app\app.py
echo.
pause
