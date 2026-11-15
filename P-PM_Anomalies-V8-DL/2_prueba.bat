@echo off
rem Corrida corta (15-30 min): comprueba que todo funciona y estima cuanto tardara la corrida completa.
cd /d "%~dp0"
.venv\Scripts\python 02_scripts\02_analysis\12_v8_dl.py --prueba %*
pause
