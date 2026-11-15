@echo off
rem Corrida completa. Si se interrumpe, se vuelve a correr este mismo archivo y sigue donde iba.
cd /d "%~dp0"
.venv\Scripts\python 02_scripts\02_analysis\12_v8_dl.py %*
pause
