@echo off
echo ========================================================
echo Starting Irodori-TTS Voice Cloning UI
echo ========================================================
echo URL: http://localhost:7860
echo Please wait for the server to start...
python -m uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
pause
