@echo off
rem Crea el entorno de Python (.venv) en esta carpeta e instala los paquetes. Se corre una sola vez.
setlocal
cd /d "%~dp0"
set "PY="
for %%v in (3.13 3.12 3.14) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; assert (3,12) <= sys.version_info[:2] <= (3,14)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo No se encontro Python 3.12, 3.13 o 3.14. Instalelo desde https://www.python.org y vuelva a correr este archivo.
  pause
  exit /b 1
)
echo Usando %PY%
%PY% -m venv .venv || goto error
.venv\Scripts\python -m pip install --upgrade pip || goto error
.venv\Scripts\python -m pip install -r requirements.txt || goto error
set "TORCH_OK="
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
  echo Hay tarjeta NVIDIA: se instala torch con CUDA.
  for %%c in (cu130 cu128 cu126) do (
    if not defined TORCH_OK (
      .venv\Scripts\python -m pip install --force-reinstall "torch==2.15.*" --index-url https://download.pytorch.org/whl/%%c --extra-index-url https://pypi.org/simple && .venv\Scripts\python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" && set "TORCH_OK=1"
    )
  )
)
if not defined TORCH_OK (
  echo Se instala torch para el procesador.
  .venv\Scripts\python -m pip install --force-reinstall "torch==2.15.*" || goto error
)
.venv\Scripts\python -c "import torch; print('torch', torch.__version__, '| tarjeta grafica:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no se usara')"
echo.
echo Listo. Ahora corra 2_prueba.bat
pause
exit /b 0
:error
echo.
echo La instalacion fallo. Copie lo que aparece arriba y envieselo a Luis.
pause
exit /b 1
