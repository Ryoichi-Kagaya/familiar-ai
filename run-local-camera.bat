@echo off

cd /d "C:\Users\Blue-\mediamtx"
start "" mediamtx.exe
timeout /t 3 >nul
start "" ffmpeg -f dshow -rtbufsize 100M -framerate 30 -video_size 640x480 -i video="Web Camera - HD" -c:v libx264 -preset ultrafast -tune zerolatency -b:v 500k -f rtsp rtsp://localhost:554/stream1
timeout /t 2 >nul
cd /d "%~dp0"
uv run familiar --gui %*
