@echo off
REM Double-click this file to check that PyTorch can use the NVIDIA GPU.
cd /d "%~dp0.."
echo Testing GPU with PyTorch...
echo.
".venv\Scripts\python.exe" -c "import torch; print('PyTorch', torch.__version__); print('GPU found:', torch.cuda.is_available()); x=torch.randn(1024,1024,device='cuda'); print('GPU matmul OK:', float((x@x).abs().mean()))"
echo.
echo ===== Take a screenshot of this window and send it to Claude =====
pause
