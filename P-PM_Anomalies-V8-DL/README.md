# V8.3 con aprendizaje profundo (P-PM_Anomalies)

Corre de nuevo el modelo V8.3 de concentraciones diarias de PM2.5, PM10, NO2 y O3 en 12 municipios de Colombia sin
monitoreo y agrega la red temporal convolucional (TCN) de Gregori como métodos adicionales, evaluados en los mismos
pliegues que los demás. Autor: Luis Miguel Hernández Beleño (Universidad del Magdalena). La descripción completa del
método está al inicio de `02_scripts/02_analysis/12_v8_dl.py`.

## Antes de correr

Las covariables de Earth Engine de la corrida V8.3 no están en el repositorio por su tamaño. Luis comparte aparte
**`resultados_v8_insumos.zip`** (303 MB); se pone, sin descomprimir, en **`archive/`**. El script comprueba que el
código, los datos y las covariables tengan la huella de la corrida V8.3 y se detiene si algo no coincide.

## Correr

Python 3.12-3.14, unos 10 GB libres y 16 GB de memoria o más.

- Windows: `1_instalar.bat` (crea `.venv` con las versiones de la V8.3; torch con CUDA si hay NVIDIA),
  `2_prueba.bat` (15-30 min; al final estima la duración de la corrida completa) y `3_corrida_completa.bat`.
- Linux o Mac: `./correr.sh instalar`, `./correr.sh prueba`, `./correr.sh completa`.
- A mano: `pip install -r requirements.txt torch==2.15.*` y `python 02_scripts/02_analysis/12_v8_dl.py [--prueba] [--cpu]`.

Al empezar mide si conviene la tarjeta gráfica (CUDA o MPS) o el procesador, cuántas redes entrenar a la vez y con
cuántos hilos, y deja dos núcleos libres; esto cambia el tiempo, no los resultados. Si se interrumpe, el mismo comando
sigue donde iba (avance en `01_data/trabajo_v8_dl/`).

## Qué enviarle a Luis

`archive/resultados_v8_dl.zip` (la última línea dice «Resultados en … verificado») y, si algo falla,
`01_data/trabajo_v8_dl/resultados*/REGISTRO.txt`.
