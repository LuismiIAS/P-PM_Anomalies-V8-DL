#!/usr/bin/env bash
# Linux o Mac. Uso:  ./correr.sh instalar   |   ./correr.sh prueba   |   ./correr.sh completa
set -e
cd "$(dirname "$0")"
case "$1" in
  instalar)
    PY=""
    for v in 3.13 3.12 3.14; do command -v "python$v" >/dev/null 2>&1 && PY="python$v" && break; done
    PY=${PY:-python3}
    "$PY" -c 'import sys; assert (3,12) <= sys.version_info[:2] <= (3,14), sys.version' \
      || { echo "Se necesita Python 3.12, 3.13 o 3.14"; exit 1; }
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install "torch==2.15.*"          # en Linux trae CUDA; en Mac usa la GPU de Apple
    if command -v nvidia-smi >/dev/null 2>&1 && ! .venv/bin/python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
      for c in cu130 cu128 cu126; do       # controlador más antiguo: se prueba una versión de CUDA anterior
        .venv/bin/python -m pip install --force-reinstall "torch==2.15.*" --index-url "https://download.pytorch.org/whl/$c" \
          --extra-index-url https://pypi.org/simple \
          && .venv/bin/python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' && break || true
      done
    fi
    .venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
    echo "Listo. Ahora: ./correr.sh prueba" ;;
  prueba)   .venv/bin/python 02_scripts/02_analysis/12_v8_dl.py --prueba "${@:2}" ;;
  completa) .venv/bin/python 02_scripts/02_analysis/12_v8_dl.py "${@:2}" ;;
  *) echo "Uso: ./correr.sh instalar | prueba | completa" ; exit 1 ;;
esac
