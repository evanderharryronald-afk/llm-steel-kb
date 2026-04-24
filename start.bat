@echo off
start "" "C:\Users\10296\AppData\Local\Programs\Ollama\ollama.exe" serve
timeout /t 3
echo Ollama started. Run your query now.