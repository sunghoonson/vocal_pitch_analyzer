@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run SETUP_ENV.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import sys, numpy, librosa, soundfile, pyqtgraph, PySide6; print('Python   :', sys.version); print('NumPy    :', numpy.__version__); print('librosa  :', librosa.__version__); print('SoundFile:', soundfile.__version__); print('pyqtgraph:', pyqtgraph.__version__); print('PySide6  :', PySide6.__version__)"

pause
