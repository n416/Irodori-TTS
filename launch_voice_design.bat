@echo off
echo ========================================================
echo Starting Irodori-TTS Voice Design UI
echo ========================================================
echo URL: http://localhost:7861
echo Please wait for the server to start...
python -m uv run python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
pause
