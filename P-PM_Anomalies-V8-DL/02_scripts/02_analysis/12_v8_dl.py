r"""
12_v8_dl.py — aprendizaje profundo (red TCN de Gregori) dentro del marco de la V8, comparado con los demás métodos

Carpeta de trabajo: E:\P-PM_Anomalies\02_scripts\02_analysis\. Copia idéntica en src/analysis/ del repositorio
compartido. Las rutas salen de 00_functions/paths.py.

QUÉ HACE
  Corre de nuevo la V8.3 completa, con los mismos datos, covariables, estaciones, pliegues de validación, regla de
  elección, intervalos y estimaciones en los municipios, y agrega como métodos la red temporal convolucional de
  Gregori (SpatialTCN de src/model/spatial_tcn_v6.py: bloques causales dilatados con conexión residual, codificador
  de variables fijas del lugar y cabeza densa). Todos los métodos se evalúan en los mismos pliegues y compiten con
  la misma regla; la red puede ganar, empatar o perder, y cualquiera de las tres cosas es un resultado.

  Métodos con la red, en el orden en que entran a la regla de elección (después de los seis de la V8):
    dl_covariables      la red solo con las covariables del lugar (meteorología, satélite, calendario y variables
                        fijas), sin estaciones vecinas: el planteamiento de la V6 y la V7.
    dl_covariables_idw  la salida de dl_covariables corregida con el inverso de la distancia al cuadrado de sus
                        residuos en las estaciones de entrenamiento de cada día (como en la regresión-kriging).
    dl_hibrido          la red con las mismas entradas que el híbrido de árboles (inverso de la distancia del día,
                        dispersión y número de vecinas, diferencias de covariables con las vecinas y covariables).
    dl_promedio         media geométrica de idw y dl_hibrido, con pesos iguales fijos.
    dl_gbm              media geométrica del híbrido de árboles y dl_hibrido, con pesos iguales fijos.

  Decisiones declaradas antes de correr:
    - Hiperparámetros: los que Gregori eligió con Optuna para cada contaminante en la V7.1 (ventana, bloques, ancho,
      núcleo, dropout, tasa de aprendizaje, decaimiento, lote, recorte del gradiente y pérdida), leídos de
      archive/resultados_v7_1/V7_OBJETIVO_SVCA/OPTUNA_V6/<objetivo>/BEST_PARAMS_V6.json. No se reajustan.
    - Entrenamiento como en la V6: AdamW, ReduceLROnPlateau (factor 0.5, paciencia max(2, paciencia/4)), recorte
      del gradiente, mejor estado por pérdida de validación, 60 épocas como máximo y paciencia de 10 (los valores
      con que Gregori eligió los hiperparámetros), promedio de 3 semillas (42, 137, 2026).
    - Cada época recorre 32 768 ventanas sacadas al azar de las filas de entrenamiento (las mismas filas que el
      híbrido de árboles: estaciones con vecinas a varios radios). Con épocas completas la corrida no cabe en un
      equipo personal.
    - La parada temprana usa el 15 % de los grupos de estaciones de entrenamiento (estaciones a menos de 20 km), que
      la red no ve: validación interna espacial dentro de cada pliegue.
    - Escalado como en la V6: estandarización con las filas de entrenamiento de las entradas dinámicas, de las
      fijas y del objetivo (logaritmo de la concentración). Los faltantes quedan en la media, con un canal
      indicador para cada variable con más de 1 % de faltantes. Los valores estandarizados se recortan a ±10 para
      que un valor extremo en una celda no dispare la red.
    - La corrección de Duan al volver de logaritmos, por clase de distancia, como en los demás métodos.
    - Los residuos de dl_covariables en las estaciones de entrenamiento son de ajuste (la red las vio), por lo que la
      corrección de dl_covariables_idw tiende a quedarse corta; su corrección de Duan se calcula dejando cada
      estación fuera de la interpolación.

ENTRADAS (las lee solo, sin descomprimir nada a mano)
  OneDrive  src/model/V8_estaciones.zip      código y datos de la V8.3
            archive/resultados_v8.zip        covariables de Earth Engine de la V8.3 y su huella (FIRMA_CORRIDA)
            archive/resultados_v7_1/V7_OBJETIVO_SVCA/OPTUNA_V6/<objetivo>/BEST_PARAMS_V6.json
  Antes de correr comprueba que el código, los datos y las covariables son los de la corrida V8.3 (misma huella).

SALIDAS (un solo lugar)
  OneDrive  archive/resultados_v8_dl.zip   todos los resultados, con el formato de resultados_v8.zip, más:
              COMPARACION_DL.csv           cada método con red frente a idw, el híbrido de árboles, la referencia
                                           sin vecinas y la mejor alternativa sin red del mismo diseño
              REPRODUCIBILIDAD_V8_3.csv    los métodos de la V8.3 recalculados frente a los archivados
              DL_ENTRENAMIENTO.csv         cada red entrenada: épocas, mejor época, pérdida, tiempo
              dl_hiperparametros.json, LEEME.txt, REGISTRO.txt
  E:        01_data\trabajo_v8_dl\         insumos extraídos, redes entrenadas y avance, para retomar si se
                                           interrumpe. Se puede mover a _to_delete cuando el zip esté en OneDrive.

USO (Símbolo del sistema, en E:\P-PM_Anomalies)
  .venv\Scripts\python -m pip install torch scikit-learn scipy psutil       (una sola vez)
  .venv\Scripts\python 02_scripts\02_analysis\12_v8_dl.py --prueba          (unos 15 min; comprueba todo y estima
                                                                              la duración de la corrida completa)
  .venv\Scripts\python 02_scripts\02_analysis\12_v8_dl.py                   (corrida completa)
  Si se interrumpe (apagado, reinicio), el mismo comando sigue donde iba: los contaminantes terminados se
  conservan y las redes ya entrenadas se cargan del avance.

USO DEL EQUIPO Y RENDIMIENTO
  Al empezar mide en este equipo cuántas redes conviene entrenar a la vez y con cuántos hilos cada una (1-2 min) y
  usa la combinación más rápida, dejando dos núcleos físicos libres. Las 6 redes de cada pliegue (2 redes x 3
  semillas) entrenan en procesos paralelos que leen los mismos datos de la memoria compartida, mientras la V8 ajusta
  sus métodos del mismo pliegue. Las tablas de cada estación se calculan una sola vez por pliegue y se reutilizan.
  Nada de esto cambia los resultados: los métodos de la V8 salen idénticos a una corrida de la V8 sola y cada red
  es reproducible con su semilla. Prioridad "por debajo de lo normal": Windows atiende primero lo que usted esté
  haciendo. Evita que el equipo se suspenda mientras corre (la pantalla sí se puede apagar). Memoria esperada
  6-10 GB; si quedan menos de 3 GB libres lo anota en el registro.
"""
import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from datetime import datetime

# ── núcleos: se fijan antes de importar numpy, scikit-learn y torch ──
NUCLEOS = max(2, (os.cpu_count() or 4) - 2)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(NUCLEOS))

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(AQUI), "00_functions"))
try:
    import paths  # noqa: E402
    REPO, LOCAL = paths.REPO, paths.LOCAL
except ImportError:                    # copia de GitHub (sin paths.py): todo dentro de la carpeta del repositorio
    RAIZ = os.path.dirname(os.path.dirname(AQUI))
    REPO = os.environ.get("P_PM_REPO", RAIZ)
    LOCAL = os.environ.get("P_PM_LOCAL", RAIZ)

VERSION_DL = "V8.3 + DL 1.0 (16-nov-2026)"
ZIP_PAQUETE = os.path.join(REPO, "src", "model", "V8_estaciones.zip")
ZIP_V83 = os.path.join(REPO, "archive", "resultados_v8.zip")
INSUMOS_V83 = os.path.join(REPO, "archive", "resultados_v8_insumos.zip")   # solo lo que usa este script (303 MB)
PARTES_V83 = os.path.join(REPO, "archive", "insumos_v8_3")                  # lo mismo, en partes de 24 MB
GREGORI = os.path.join(REPO, "archive", "resultados_v7_1", "V7_OBJETIVO_SVCA", "OPTUNA_V6")
ZIP_FINAL = os.path.join(REPO, "archive", "resultados_v8_dl.zip")
TRABAJO = os.path.join(LOCAL, "01_data", "trabajo_v8_dl")

OBJETIVO_V7 = {"pm25": "PM25_ugm3", "pm10": "PM10_ugm3", "no2": "NO2_ugm3", "o3": "O3MDA8_ugm3"}
SEMILLAS = (42, 137, 2026)            # las de la V6
EPOCAS_MAX, PACIENCIA = 60, 10        # optuna_epochs y optuna_patience de la V6
MUESTRAS_EPOCA = 32768                # ventanas por época
MUESTRAS_VALIDACION = 16384           # ventanas fijas de la validación interna
FRACCION_VALIDACION = 0.15            # fracción de grupos de estaciones para la parada temprana
RECORTE = 10.0                        # recorte de los valores estandarizados
DL_METODOS = ["dl_covariables", "dl_covariables_idw", "dl_hibrido", "dl_promedio", "dl_gbm"]
DESC_DL = {"dl_covariables": "red TCN solo con covariables del lugar, sin estaciones vecinas",
           "dl_covariables_idw": "dl_covariables corregida con el inverso de la distancia de sus residuos en las estaciones",
           "dl_hibrido": "red TCN con las entradas del híbrido (inverso de la distancia del día, vecinas y covariables)",
           "dl_promedio": "media geométrica de idw y dl_hibrido, con pesos iguales fijos",
           "dl_gbm": "media geométrica del híbrido de árboles y dl_hibrido, con pesos iguales fijos"}
CALENDARIO = ("doy_sin", "doy_cos", "dia_semana")


# ═══════════════════════════════════════════════════════════════════
# PREPARACIÓN
# ═══════════════════════════════════════════════════════════════════
def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as x:
        for b in iter(lambda: x.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


HUELLA_SCRIPT = sha256(os.path.abspath(__file__))     # una red guardada solo se reutiliza con este mismo script


def comprobar_paquetes():
    faltan = []
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "psutil"):
        try:
            __import__(mod)
        except ImportError:
            faltan.append({"sklearn": "scikit-learn"}.get(mod, mod))
    if faltan:
        print("FALTAN PAQUETES. En E:\\P-PM_Anomalies, una sola vez:\n"
              f"  .venv\\Scripts\\python -m pip install {' '.join(faltan)}")
        sys.exit(1)


def preparar_equipo():
    """Prioridad baja y sin suspensión mientras corre."""
    import psutil
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception as e:
        print(f"(no se pudo bajar la prioridad: {e})")
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # continuo + sistema
        except Exception:
            pass
    mem = psutil.virtual_memory()
    return (f"Equipo: {os.cpu_count()} núcleos lógicos; la V8 usa {NUCLEOS}; prioridad baja. "
            f"Memoria {mem.total / 2**30:.0f} GB, libre {mem.available / 2**30:.0f} GB")


def unir_partes(log):
    """En la copia de GitHub, los insumos de la V8.3 vienen en partes de menos de 25 MB (el límite de GitHub); se
    unen en TRABAJO y se comprueban con su SHA-256 antes de usarlos."""
    man = json.load(open(os.path.join(PARTES_V83, "MANIFIESTO.json"), encoding="utf-8"))
    destino = os.path.join(TRABAJO, man["archivo"])
    if os.path.exists(destino) and os.path.getsize(destino) == man["bytes"] and sha256(destino) == man["sha256"]:
        return destino
    log(f"Uniendo {len(man['partes'])} partes de {man['archivo']} ...")
    with open(destino + ".tmp", "wb") as out:
        for p in man["partes"]:
            ruta = os.path.join(PARTES_V83, p["nombre"])
            if not os.path.exists(ruta) or sha256(ruta) != p["sha256"]:
                log(f"NO SE PUEDE CORRER: la parte {p['nombre']} falta o está incompleta; vuelva a descargar el repositorio")
                sys.exit(1)
            with open(ruta, "rb") as x:
                shutil.copyfileobj(x, out, 1 << 24)
    if sha256(destino + ".tmp") != man["sha256"]:
        log("NO SE PUEDE CORRER: el archivo unido no coincide con el original")
        sys.exit(1)
    os.replace(destino + ".tmp", destino)
    return destino


def extraer_insumos(log):
    """Código y datos del paquete V8.3 y covariables de la corrida V8.3 en TRABAJO, con la huella comprobada."""
    global ZIP_V83
    os.makedirs(TRABAJO, exist_ok=True)
    if not os.path.exists(ZIP_V83):
        if os.path.exists(INSUMOS_V83):
            ZIP_V83 = INSUMOS_V83
        elif os.path.exists(os.path.join(PARTES_V83, "MANIFIESTO.json")):
            ZIP_V83 = unir_partes(log)
        else:
            log(f"NO SE PUEDE CORRER: faltan las covariables de la V8.3. Ponga resultados_v8_insumos.zip (o "
                f"resultados_v8.zip) en {os.path.dirname(INSUMOS_V83)}")
            sys.exit(1)
    if not os.path.exists(ZIP_PAQUETE):
        log(f"NO SE PUEDE CORRER: falta {ZIP_PAQUETE}")
        sys.exit(1)
    cod = os.path.join(TRABAJO, "V8_estaciones")
    with zipfile.ZipFile(ZIP_PAQUETE) as z:
        for n in z.namelist():
            destino = os.path.join(TRABAJO, n)
            if n.endswith("/") or (os.path.exists(destino) and os.path.getsize(destino) == z.getinfo(n).file_size):
                continue
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            with z.open(n) as src, open(destino, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 24)
    gee = os.path.join(cod, "gee")
    os.makedirs(gee, exist_ok=True)
    with zipfile.ZipFile(ZIP_V83) as z:
        for n in ("dinamicas.csv", "fijas.csv", "cabeceras.csv", "rejilla.csv"):
            ruta = os.path.join(gee, n)
            if not (os.path.exists(ruta) and os.path.getsize(ruta) == z.getinfo(f"gee/{n}").file_size):
                log(f"Extrayendo gee/{n} de la corrida V8.3 ...")
                with z.open(f"gee/{n}") as src, open(ruta, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
        firma_v83 = z.read("resultados/FIRMA_CORRIDA.txt").decode().strip()
        resumen_v83 = z.read("resultados/VALIDACION_RESUMEN.csv")
    sys.path.insert(0, cod)
    import correr_v8 as c8
    firma = c8.firma_corrida()
    if firma != firma_v83:
        log("NO SE PUEDE CORRER: el código, los datos o las covariables no son los de la corrida V8.3 "
            f"(huella {firma[:12]} frente a {firma_v83[:12]}).")
        sys.exit(1)
    log(f"Código, datos y covariables idénticos a la corrida V8.3 (huella {firma[:16]})")
    return c8, firma, resumen_v83


def leer_hiperparametros(log):
    hp = {}
    for c, obj in OBJETIVO_V7.items():
        f = os.path.join(GREGORI, obj, "BEST_PARAMS_V6.json")
        if not os.path.exists(f):
            log(f"NO SE PUEDE CORRER: falta {f}")
            sys.exit(1)
        p = json.load(open(f, encoding="utf-8"))["params"]
        hp[c] = {k: p[k] for k in ("window_days", "n_blocks", "channel_width", "kernel_size", "dropout",
                                   "learning_rate", "weight_decay", "batch_size", "gradient_clip", "loss")}
        log(f"  {c}: {hp[c]}")
    return hp


# ═══════════════════════════════════════════════════════════════════
# RED (SpatialTCN de la V6, con una salida: el valor del día)
# ═══════════════════════════════════════════════════════════════════
def construir_red(torch):
    nn = torch.nn

    class Chomp1d(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return x if self.n == 0 else x[:, :, :-self.n].contiguous()

    class TemporalBlock(nn.Module):
        def __init__(self, nin, nout, kernel, dilation, dropout):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(nin, nout, kernel, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(nout, nout, kernel, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout))
            self.res = nn.Conv1d(nin, nout, 1) if nin != nout else nn.Identity()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.net(x) + self.res(x))

    class SpatialTCN(nn.Module):
        def __init__(self, n_dynamic, n_static, channels, kernel_size, dropout):
            super().__init__()
            self.tcn = nn.Sequential(*[TemporalBlock(n_dynamic if i == 0 else channels[i - 1], ch, kernel_size,
                                                     2 ** i, dropout) for i, ch in enumerate(channels)])
            sh = max(8, min(64, n_static * 4))
            self.static_encoder = nn.Sequential(nn.Linear(n_static, sh), nn.ReLU(), nn.Dropout(dropout))
            self.head = nn.Sequential(nn.Linear(channels[-1] + sh, 128), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(128, 1))

        def forward(self, x, s):                      # x: lote x ventana x canales; s: lote x fijas
            zt = self.tcn(x.transpose(1, 2))[:, :, -1]
            return self.head(torch.cat([zt, self.static_encoder(s)], 1)).squeeze(1)

    return SpatialTCN


class Escalador:
    """Estandarización con las filas de entrenamiento, faltantes en la media con canal indicador (variables con más
    de 1 % de faltantes) y recorte a ±RECORTE."""

    def __init__(self, X):
        import numpy as np
        self.mu = np.nanmean(X, 0).astype("float32")
        self.sd = np.nanstd(X, 0).astype("float32")
        self.mu[~np.isfinite(self.mu)] = 0
        self.sd[~(self.sd > 0)] = 1
        self.ind = np.where(np.isnan(X).mean(0) > 0.01)[0]

    @property
    def canales(self):
        return len(self.mu) + len(self.ind)

    def transformar(self, A, out=None, lote=64):
        """A: (..., variables) -> (..., variables + indicadores), por lotes del primer eje para no duplicar memoria.
        Con `out`, escribe ahí (memoria compartida con los procesos que entrenan)."""
        import numpy as np
        if out is None:
            out = np.empty(A.shape[:-1] + (self.canales,), dtype="float32")
        for k in range(0, A.shape[0], lote):
            z = (A[k:k + lote] - self.mu) / self.sd
            falt = np.isnan(z)
            out[k:k + lote, ..., :len(self.mu)] = np.clip(np.nan_to_num(z), -RECORTE, RECORTE)
            if len(self.ind):
                out[k:k + lote, ..., len(self.mu):] = falt[..., self.ind]
        return out


# ── entrenamiento y predicción de una red (funciones de módulo: las corren los procesos de trabajo) ──
def dispositivo_disponible():
    """'cuda' (tarjeta NVIDIA), 'mps' (Mac con chip Apple) o 'cpu'."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configurar_torch(hilos, disp):
    """Hilos y precisión: en tarjeta gráfica se fuerza float32 completo (sin TF32) y convoluciones deterministas,
    para que cada semilla sea reproducible."""
    import torch
    torch.set_num_threads(hilos)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if disp == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def nueva_red(torch, hp, n_din, n_fijas):
    return construir_red(torch)(n_din, n_fijas, [int(hp["channel_width"])] * int(hp["n_blocks"]),
                                int(hp["kernel_size"]), float(hp["dropout"]))


def ventanas(A, b, t, L):
    """A: tensor (bloques, días + L - 1, canales); b, t: tensores de índices. Devuelve (lote, L, canales)."""
    import torch
    return A[b[:, None], t[:, None] + torch.arange(L, device=A.device)[None, :]]


def ajustar_red(red, Ad, As, b, t, ys, es_val, L, hp, semilla, epocas, paciencia):
    """Entrenamiento de la V6: AdamW, ReduceLROnPlateau, recorte del gradiente y mejor estado por la pérdida de
    validación, con parada temprana. Ad, As, b, t, ys son tensores en el dispositivo de la red; es_val, numpy."""
    import numpy as np
    import torch
    nn = torch.nn
    dev = Ad.device
    rng = np.random.default_rng(semilla)
    opt = torch.optim.AdamW(red.parameters(), lr=float(hp["learning_rate"]), weight_decay=float(hp["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(2, paciencia // 4))
    crit = {"mse": nn.MSELoss(), "mae": nn.L1Loss()}.get(hp["loss"], nn.SmoothL1Loss(beta=1.0))
    tr = np.where(~es_val)[0]
    va = np.where(es_val)[0]
    if len(va) > MUESTRAS_VALIDACION:
        va = np.sort(rng.choice(va, MUESTRAS_VALIDACION, replace=False))
    va_t = torch.from_numpy(va).to(dev)
    bs = int(hp["batch_size"])
    mejor, estado, espera, mejor_ep, ep, perdidas = np.inf, None, 0, 0, 0, [np.nan]
    for ep in range(1, epocas + 1):
        red.train()
        sel = torch.from_numpy(rng.choice(tr, min(len(tr), MUESTRAS_EPOCA), replace=False)).to(dev)
        suma, n = torch.zeros((), device=dev), 0
        for k in range(0, len(sel), bs):
            i = sel[k:k + bs]
            opt.zero_grad(set_to_none=True)
            loss = crit(red(ventanas(Ad, b[i], t[i], L), As[b[i]]), ys[i])
            loss.backward()
            nn.utils.clip_grad_norm_(red.parameters(), float(hp["gradient_clip"]))
            opt.step()
            suma += loss.detach()
            n += 1
        perdidas = [float(suma) / max(n, 1)]
        if len(va):
            vl = float(crit(predecir_red([red], Ad, As, b[va_t], t[va_t], L, numpy=False), ys[va_t]))
        else:
            vl = perdidas[0]
        sched.step(vl)
        if np.isfinite(vl) and vl < mejor - 1e-7:
            mejor, mejor_ep, espera = vl, ep, 0
            estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
        else:
            espera += 1
            if espera >= paciencia:
                break
    if estado is None:
        raise RuntimeError("la red no produjo una pérdida de validación finita")
    red.load_state_dict(estado)
    return {"epocas": ep, "mejor_epoca": mejor_ep, "perdida_validacion": round(mejor, 6),
            "perdida_entrenamiento_ultima": round(perdidas[0], 6)}


def predecir_red(redes, Ad, As, b, t, L, lote=8192, numpy=True):
    """Promedio de las redes, en unidades estandarizadas. Ad, As: tensores; b, t: tensores o arreglos."""
    import torch
    dev = Ad.device
    b = torch.as_tensor(b, device=dev)
    t = torch.as_tensor(t, device=dev)
    out = torch.zeros(len(b), device=dev)
    with torch.no_grad():
        for red in redes:
            red.eval()
            for k in range(0, len(b), lote):
                out[k:k + lote] += red(ventanas(Ad, b[k:k + lote], t[k:k + lote], L), As[b[k:k + lote]])
    out /= len(redes)
    return out.cpu().numpy() if numpy else out


def _inicio_trabajador(hilos, disp):
    """Cada proceso de trabajo: sus hilos, su precisión y prioridad baja."""
    configurar_torch(hilos, disp)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def _abrir_compartida(nombre):
    from multiprocessing import shared_memory
    kw = {"track": False} if sys.version_info >= (3, 13) else {}
    return shared_memory.SharedMemory(name=nombre, **kw)


def _trabajo_red(tarea):
    """Una red (una semilla): la entrena, o la carga del avance si ya estaba, y predice las filas pedidas.
    Las ventanas se leen de la memoria compartida; en tarjeta gráfica se copian a ella una vez."""
    import numpy as np
    import torch
    disp = tarea["disp"]
    shm = _abrir_compartida(tarea["memoria"])
    A = np.ndarray(tarea["forma"], dtype=np.float32, buffer=shm.buf)
    Ad = None
    try:
        Ad = torch.from_numpy(A) if disp == "cpu" else torch.from_numpy(A).to(disp)
        a = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(disp)          # noqa: E731
        As, b, t, ys = a(tarea["As"]), a(tarea["b"].astype(np.int64)), a(tarea["t"].astype(np.int64)), a(tarea["ys"])
        hp, L, ruta = tarea["hp"], tarea["L"], tarea["ruta"]
        torch.manual_seed(tarea["semilla"])
        red = nueva_red(torch, hp, A.shape[2], tarea["As"].shape[1]).to(disp)
        if os.path.exists(ruta) and os.path.exists(ruta + ".json"):
            red.load_state_dict(torch.load(ruta, map_location=disp))
            h = json.load(open(ruta + ".json"))
        else:
            t1 = time.time()
            h = ajustar_red(red, Ad, As, b, t, ys, tarea["es_val"], L, hp, tarea["semilla"], tarea["epocas"],
                            tarea["paciencia"])
            h["segundos"] = round(time.time() - t1, 1)
            h["dispositivo"] = disp
            torch.save({k: v.cpu() for k, v in red.state_dict().items()}, ruta + ".tmp")
            os.replace(ruta + ".tmp", ruta)
            json.dump(h, open(ruta + ".json", "w"))
        pred = predecir_red([red], Ad, As, tarea["pred_b"].astype(np.int64), tarea["pred_t"].astype(np.int64), L)
        return {"h": h, "pred": pred}
    finally:
        del Ad, A
        shm.close()


def _listo(disp):
    import torch
    if disp != "cpu":
        torch.zeros(1, device=disp)
    return os.getpid()


def _banco(args):
    """Rendimiento de entrenamiento con la red de PM2.5 y datos sintéticos, para elegir procesos e hilos."""
    import torch
    inicio, duracion, hp, disp = args
    torch.manual_seed(0)
    red = nueva_red(torch, hp, 60, 26).to(disp)
    opt = torch.optim.AdamW(red.parameters(), lr=1e-3)
    bs = int(hp["batch_size"])
    A = torch.randn(400, 900, 60, device=disp)
    S = torch.randn(400, 26, device=disp)
    y = torch.randn(bs, device=disp)
    while time.time() < inicio:
        time.sleep(0.01)
    n, t0 = 0, time.time()
    while time.time() - t0 < duracion:
        b = torch.randint(0, 400, (bs,), device=disp)
        t = torch.randint(0, 880, (bs,), device=disp)
        opt.zero_grad()
        ((red(ventanas(A, b, t, int(hp["window_days"])), S[b]) - y) ** 2).mean().backward()
        opt.step()
        n += bs
    if disp == "cuda":
        torch.cuda.synchronize()
    return n, t0, time.time()


def afinar(hp_pm25, log, forzar_cpu=False):
    """Elige el dispositivo (tarjeta gráfica si hay y es más rápida), cuántas redes entrenar a la vez y con cuántos
    hilos cada una, midiendo en este equipo. El resultado del modelo no depende de esta elección; solo el tiempo."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import psutil
    fisicos = psutil.cpu_count(logical=False) or NUCLEOS
    total = max(2, min(NUCLEOS, fisicos - 2))
    libre_gb = psutil.virtual_memory().available / 2**30
    max_proc = max(1, int((libre_gb - 4) / 0.6))           # ~0.6 GB por proceso de trabajo
    disp = "cpu" if forzar_cpu else dispositivo_disponible()
    candidatos = []
    if disp != "cpu":
        import torch
        if disp == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30
            nombre = torch.cuda.get_device_name(0)
            max_gpu = max(1, int(vram * 0.8 / 1.2))          # ~1.2 GB de tarjeta por red (datos + contexto)
        else:
            vram, nombre, max_gpu = 0, "Apple", 2
        log(f"  tarjeta gráfica: {nombre}" + (f" ({vram:.0f} GB)" if vram else ""))
        candidatos += [(p, max(1, min(4, total // p)), disp) for p in (1, 2, 3, 4, 6, 8, 12)
                       if p <= min(max_gpu, max_proc, total)]
    candidatos += [(p, max(1, total // p), "cpu") for p in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
                   if p <= min(total, max_proc)]
    medidas, tope = [], {}
    for p, h, d in candidatos:
        if p > tope.get(d, 10**6):
            continue
        try:
            with ProcessPoolExecutor(p, mp_context=mp.get_context("spawn"), initializer=_inicio_trabajador,
                                     initargs=(h, d)) as ex:
                list(ex.map(_listo, [d] * p))
                r = list(ex.map(_banco, [(time.time() + 3, 10.0, hp_pm25, d)] * p))
            rend = sum(x[0] for x in r) / (max(x[2] for x in r) - min(x[1] for x in r))
        except Exception as e:
            log(f"  {p} redes a la vez x {h} hilos en {d}: falló ({type(e).__name__}); se descarta")
            continue
        medidas.append((p, h, d, rend))
        log(f"  {p} redes a la vez x {h} hilos en {d}: {rend:.0f} ventanas/s")
        if len([m for m in medidas if m[2] == d]) >= 2 and rend < 0.9 * max(m[3] for m in medidas if m[2] == d):
            tope[d] = p          # más procesos ya no ayudan en este dispositivo: no se prueban más
    mejor = max(x[3] for x in medidas)
    p, h, d, rend = min((x for x in medidas if x[3] >= 0.97 * mejor), key=lambda x: (x[2] == "cpu", x[0]))
    log(f"  elegido: {p} redes a la vez con {h} hilos cada una, en {d} ({fisicos} núcleos físicos, "
        f"{libre_gb:.0f} GB de memoria libre)")
    return p, h, d, rend


# ═══════════════════════════════════════════════════════════════════
# ENTRENAMIENTO Y PREDICCIÓN DENTRO DEL MARCO DE LA V8
# ═══════════════════════════════════════════════════════════════════
class DL:
    def __init__(self, m, hp, cache, log, prueba, procesos, hilos, disp):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        import numpy as np
        import torch
        self.np, self.torch, self.m, self.hp, self.cache, self.log = np, torch, m, hp, cache, log
        self.semillas = SEMILLAS
        self.epocas = 3 if prueba else EPOCAS_MAX
        self.paciencia = 2 if prueba else PACIENCIA
        self.historial = []
        self.n_ajuste = 0
        os.makedirs(cache, exist_ok=True)
        self.disp = disp
        configurar_torch(max(1, min(NUCLEOS, procesos * hilos)), disp)    # predicción en el proceso principal
        # pliegues en vuelo: cada pliegue da 6 redes; con más procesos que eso se adelantan los pliegues siguientes
        self.profundidad = max(1, -(-procesos // (2 * len(SEMILLAS))))
        self.pool = ProcessPoolExecutor(procesos, mp_context=mp.get_context("spawn"),
                                        initializer=_inicio_trabajador, initargs=(hilos, disp))

    def cerrar(self):
        self.pool.shutdown()

    # ── columnas de cada red ──
    def columnas(self, cols, dat, variante):
        fijas = [c for c in cols if c in dat.fix_cols]
        if variante == "dl_covariables":
            din = [c for c in cols if c in dat.dyn_cols or c in CALENDARIO]
        else:
            fuera = set(self.m.COLS_SUELO) | set(self.m.COLS_PLACEBO) | set(fijas)
            din = [c for c in cols if c not in fuera and not c.startswith("du_")]
        return din, fijas

    def _arreglo(self, tablas, cols, L):
        """(bloques, L - 1 + T, columnas): cada bloque es un lugar con sus vecinas; L - 1 días faltantes al inicio
        para que la ventana del primer día exista."""
        np = self.np
        T = len(tablas[0])
        A = np.full((len(tablas), T + L - 1, len(cols)), np.nan, dtype="float32")
        for i, t in enumerate(tablas):
            A[i, L - 1:] = t.reindex(columns=cols).values.astype("float32")
        return A

    def _fijas(self, tablas, cols):
        np = self.np
        if not cols:
            return np.zeros((len(tablas), 1), "float32")
        return np.stack([t.reindex(columns=cols).values[0].astype("float32") for t in tablas])

    def _grupos_validacion(self, dat, train):
        """Grupos espaciales (20 km) de estaciones de entrenamiento para la parada temprana, en orden aleatorio fijo,
        hasta reunir cerca del 15 % de las estaciones; un grupo que pasaría de esa fracción se salta, para que un
        grupo grande no le quite a la red una parte desproporcionada de sus datos."""
        np = self.np
        g = dat.grupos[train]
        ug, n = np.unique(g, return_counts=True)
        meta = FRACCION_VALIDACION * len(train)
        elegidos, total = [], 0
        for k in np.random.default_rng(self.m.SEMILLA).permutation(len(ug)):
            if total + n[k] <= meta:
                elegidos.append(ug[k])
                total += n[k]
        if not elegidos and len(ug) >= 4:
            elegidos = [ug[np.argmin(n)]]
        return np.array(elegidos)

    # ── las dos redes de un pliegue (o del modelo final), con sus 3 semillas en paralelo ──
    def lanzar(self, F, dat, train):
        """Prepara los datos de las dos redes y deja sus 6 entrenamientos (2 redes x 3 semillas) corriendo en los
        procesos de trabajo; mientras tanto la V8 ajusta sus métodos. `recoger` espera y arma el resultado."""
        from multiprocessing import shared_memory
        np = self.np
        p, hp = dat.p, self.hp[dat.p]
        L = int(hp["window_days"])
        T = len(dat.fechas)
        train = np.asarray(train)
        self.n_ajuste += 1
        t0 = time.time()
        conj, futuros = {}, []
        try:
            for variante in ("dl_covariables", "dl_hibrido"):
                con_vec = variante == "dl_hibrido"
                if con_vec:
                    Fv = F                                  # las mismas filas que el híbrido de árboles
                    claves = list(F[["sid_idx", "radio"]].drop_duplicates().itertuples(index=False, name=None))
                else:
                    Fv = dat.filas_entrenamiento(train, np.random.default_rng(self.m.SEMILLA), False)
                    claves = [(int(s), 0) for s in train]

                def hacer(k, con_vec=con_vec):
                    s, r = int(k[0]), int(k[1])
                    vec = np.array([u for u in train if u != s and dat.D[s, u] > max(r, 0.5)], dtype=np.int64)
                    return dat.variables(s, dat.D[s], vec, con_vec)
                tablas = self.m.mapa(hacer, claves)
                din, fijas = self.columnas(list(tablas[0].columns), dat, variante)
                pos = {(int(s), int(r)): i for i, (s, r) in enumerate(claves)}
                radio = Fv["radio"].values if "radio" in Fv else np.zeros(len(Fv), int)
                b = np.array([pos[(int(s), int(r))] for s, r in zip(Fv["sid_idx"].values, radio)], dtype=np.int32)
                t = Fv["t"].values.astype(np.int32)
                y = Fv["y"].values.astype("float32")
                es_val = np.isin(dat.grupos[Fv["sid_idx"].values], self._grupos_validacion(dat, train))
                Araw = self._arreglo(tablas, din, L)
                del tablas
                Sraw = np.stack([dat.F[fijas].values[s] for s, _ in claves]).astype("float32") if fijas else \
                    np.zeros((len(claves), 1), "float32")
                esc_d = Escalador(Araw[b[~es_val], t[~es_val] + L - 1])
                esc_s = Escalador(Sraw[np.unique(b[~es_val])])
                mu_y, sd_y = float(y[~es_val].mean()), float(y[~es_val].std() or 1.0)
                forma = Araw.shape[:2] + (esc_d.canales,)
                shm = shared_memory.SharedMemory(create=True, size=int(np.prod(forma)) * 4)
                Ad = np.ndarray(forma, dtype=np.float32, buffer=shm.buf)
                esc_d.transformar(Araw, out=Ad)
                del Araw
                As = esc_s.transformar(Sraw)
                if con_vec:
                    pred_b, pred_t = b, t                   # filas de entrenamiento, para la corrección de Duan
                else:
                    pred_b = np.repeat(np.arange(len(claves), dtype=np.int32), T)
                    pred_t = np.tile(np.arange(T, dtype=np.int32), len(claves))   # todos los días: residuos
                clave = hashlib.sha256(json.dumps([HUELLA_SCRIPT, p, variante, train.tolist(), int(len(Fv)),
                                                   float(y.sum()), din, fijas, hp, self.epocas, self.paciencia,
                                                   MUESTRAS_EPOCA]).encode()).hexdigest()[:24]
                conj[variante] = {"shm": shm, "Ad": Ad, "forma": forma, "As": As, "b": b, "t": t, "y": y,
                                  "es_val": es_val, "din": din, "fijas": fijas, "esc_d": esc_d, "esc_s": esc_s,
                                  "mu_y": mu_y, "sd_y": sd_y, "con_vec": con_vec, "clave": clave,
                                  "dmin": Fv["dmin"].values if con_vec else None, "n": len(claves)}
                for semilla in self.semillas:
                    tarea = {"memoria": shm.name, "forma": forma, "As": As, "b": b, "t": t, "ys": (y - mu_y) / sd_y,
                             "es_val": es_val, "L": L, "hp": hp, "semilla": semilla, "epocas": self.epocas,
                             "paciencia": self.paciencia, "pred_b": pred_b, "pred_t": pred_t, "disp": self.disp,
                             "ruta": os.path.join(self.cache, f"{p}_{variante}_{clave}_{semilla}.pt")}
                    futuros.append((variante, semilla, tarea["ruta"], self.pool.submit(_trabajo_red, tarea)))
        except BaseException:
            self._liberar(conj, futuros)
            raise
        return {"conj": conj, "futuros": futuros, "dat": dat, "train": train, "t0": t0, "L": L, "T": T}

    def _liberar(self, conj, futuros):
        for *_, f in futuros:
            f.cancel()
        for *_, f in futuros:
            try:
                f.result()
            except Exception:
                pass
        for c in conj.values():
            c.pop("Ad", None)
            c["shm"].close()
            try:
                c["shm"].unlink()
            except FileNotFoundError:
                pass

    def recoger(self, pend):
        np, torch = self.np, self.torch
        conj, futuros, dat, train, L, T = (pend[k] for k in ("conj", "futuros", "dat", "train", "L", "T"))
        p, hp = dat.p, self.hp[dat.p]
        try:
            res = [(v, s, ruta, f.result()) for v, s, ruta, f in futuros]
        finally:
            self._liberar(conj, [])
        out = {"train": train}
        for variante, c in conj.items():
            redes, preds = [], []
            for v, semilla, ruta, r in res:
                if v != variante:
                    continue
                red = nueva_red(torch, hp, c["forma"][2], c["As"].shape[1]).to(self.disp)
                red.load_state_dict(torch.load(ruta, map_location=self.disp))
                red.eval()
                redes.append(red)
                preds.append(r["pred"])
                self.historial.append({"contaminante": p, "ajuste": self.n_ajuste, "red": variante, "semilla": semilla,
                                       "estaciones_entrenamiento": len(train), "filas": int(len(c["y"])),
                                       "filas_validacion_interna": int(c["es_val"].sum()), "canales": int(c["forma"][2]),
                                       "fijas": len(c["fijas"]), **r["h"]})
            pred = np.mean(preds, 0).astype(float) * c["sd_y"] + c["mu_y"]
            d = {"redes": redes, "din": c["din"], "fijas": c["fijas"], "esc_d": c["esc_d"], "esc_s": c["esc_s"],
                 "L": L, "mu_y": c["mu_y"], "sd_y": c["sd_y"], "con_vec": c["con_vec"]}
            if c["con_vec"]:
                d["smear"] = self.m.duan(c["y"] - pred, c["dmin"])
            else:
                P = pred.reshape(c["n"], T)                             # estaciones de entrenamiento x días
                d["smear"] = self.m.duan(c["y"] - P[c["b"], c["t"]], None)
                self._residuos_idw(d, P.T, dat, train)
            out[variante] = d
        self.log(f"      redes: dl_covariables {len(conj['dl_covariables']['y'])} filas, dl_hibrido "
                 f"{len(conj['dl_hibrido']['y'])} filas ({conj['dl_hibrido']['forma'][2]} canales), "
                 f"{len(self.semillas)} semillas, {time.time() - pend['t0']:.0f} s (en paralelo con la V8)")
        return out

    # ── corrección por inverso de la distancia de los residuos de dl_covariables ──
    def _residuos_idw(self, d, P, dat, train):
        """Residuos de la red (P, días x estaciones) en las estaciones de entrenamiento, y su corrección de Duan
        dejando cada estación fuera de la interpolación."""
        np = self.np
        R = dat.Y[:, train] - P
        d["R"], d["train_R"] = R.astype("float32"), train
        corr, dmin = [], []
        for k, s in enumerate(train):
            c, dm = self._idw_residuo(d, dat.D[s], excluir=k)
            corr.append(c)
            dmin.append(dm)
        res = (R - np.column_stack(corr)).ravel()
        dm = np.column_stack(dmin).ravel()
        ok = np.isfinite(res)
        d["smear_idw"] = self.m.duan(res[ok], dm[ok])

    def _idw_residuo(self, d, dist, excluir=None):
        """IDW (1/d², d >= 1 km) de los residuos del día en las estaciones de entrenamiento a más de 0.5 km del
        lugar, y la distancia a la más cercana con residuo ese día."""
        np = self.np
        R, train = d["R"], d["train_R"]
        dd = np.maximum(dist[train], 1.0).astype(float)
        usar = dist[train] > 0.5
        if excluir is not None:
            usar[excluir] = False
        m = np.isfinite(R) & usar[None, :]
        w = m / dd[None, :] ** 2
        sw = w.sum(1)
        c = np.where(sw > 0, (w * np.nan_to_num(R)).sum(1) / np.where(sw > 0, sw, 1), 0.0)
        dm = np.where(m, dd[None, :], np.inf).min(1)
        return c, np.where(np.isfinite(dm), dm, np.nan)

    # ── predicción para estaciones de prueba o puntos de los municipios ──
    def predecir_log(self, D, F, dat):
        """Logaritmos de dl_covariables, dl_covariables_idw y dl_hibrido para las filas F: estaciones de prueba (con
        sid_idx) o puntos de un municipio (bloques consecutivos de T filas, con la columna _punto)."""
        np = self.np
        T = len(dat.fechas)
        train = D["train"]
        if "sid_idx" in F:
            lugares = list(dict.fromkeys(F["sid_idx"].values.tolist()))
            pos = {s: i for i, s in enumerate(lugares)}
            b = np.array([pos[s] for s in F["sid_idx"].values])
            t = F["t"].values.astype(int)
            dist = {s: dat.D[s] for s in lugares}

            def tabla(s, con_vec):
                return dat.variables(int(s), dat.D[s], train[dat.D[s, train] > 0.5], con_vec)
            tablas = {v: self.m.mapa(lambda s, cv=D[v]["con_vec"]: tabla(s, cv), lugares)
                      for v in ("dl_covariables", "dl_hibrido")}
        else:
            puntos = F["_punto"].values.reshape(-1, T)[:, 0]
            lugares = list(puntos)
            b = np.repeat(np.arange(len(puntos)), T)
            t = np.tile(np.arange(T), len(puntos))
            dist = {i: dat.DM[i] for i in lugares}
            bloques = [F.iloc[k * T:(k + 1) * T] for k in range(len(puntos))]
            tablas = {"dl_covariables": bloques, "dl_hibrido": bloques}
        L = {}
        for v in ("dl_covariables", "dl_hibrido"):
            d = D[v]
            Ad = self.torch.from_numpy(d["esc_d"].transformar(self._arreglo(tablas[v], d["din"], d["L"]))).to(self.disp)
            As = self.torch.from_numpy(d["esc_s"].transformar(self._fijas(tablas[v], d["fijas"]))).to(self.disp)
            base = predecir_red(d["redes"], Ad, As, b.astype(np.int64), t.astype(np.int64), d["L"]).astype(float) \
                * d["sd_y"] + d["mu_y"]
            del Ad, As
            dmin = F["dmin"].values if d["con_vec"] else None
            L[v] = base + np.log(self.m.factor(d["smear"], dmin, len(F)))
            if v == "dl_covariables":
                cor = np.empty(len(F))
                dmr = np.empty(len(F))
                for k, lugar in enumerate(lugares):
                    c, dm = self._idw_residuo(d, dist[lugar])
                    sel = b == k
                    cor[sel] = c[t[sel]]
                    dmr[sel] = dm[t[sel]]
                L["dl_covariables_idw"] = base + cor + np.log(self.m.factor(d["smear_idw"], dmr, len(F)))
        return L


def instalar_dl(m, dl):
    """Agrega las redes a la V8 sin cambiar su código: los métodos nuevos pasan por la misma validación, regla de
    elección, intervalos, área de aplicabilidad y estimación en los municipios que los demás."""
    import numpy as np
    m.METODOS = list(m.METODOS) + DL_METODOS
    m.DESCRIPCION.update(DESC_DL)
    original_ajustar, original_log = m.ajustar_metodos, m.log_metodos
    original_mun = m.Datos.filas_municipio
    original_var, original_ent = m.Datos.variables, m.Datos.filas_entrenamiento

    def variables(self, j, dist, vecinas, con_vecinas=True):
        """Las tablas de las estaciones del pliegue se calculan una vez y se reutilizan (la V8 y las redes piden las
        mismas). Devuelve una copia, porque quien la pide le agrega columnas; los valores son idénticos."""
        if j >= len(self.est):
            return original_var(self, j, dist, vecinas, con_vecinas)
        memo = self.__dict__.setdefault("_memo_dl", {})
        k = (int(j), bool(con_vecinas), np.asarray(vecinas, dtype=np.int64).tobytes() if con_vecinas else b"")
        X = memo.get(k)
        if X is None:
            X = memo[k] = original_var(self, j, dist, vecinas, con_vecinas)
        return X.copy()

    def filas_entrenamiento(self, train, rng, con_vecinas=True):
        if con_vecinas:                     # empieza un pliegue (o el modelo final): se libera la memoria del anterior
            self._memo_dl = {}
        return original_ent(self, train, rng, con_vecinas)

    def ajustar_metodos(F, dat, train, n_perm=0):
        pend = dl.lanzar(F, dat, train)            # las redes entrenan mientras la V8 ajusta sus métodos
        try:
            M = original_ajustar(F, dat, train, n_perm)
        finally:
            D = dl.recoger(pend)
        M["dl"], M["dl_dat"] = D, dat
        return M

    def con_redes(L, D, F, dat):
        L.update(dl.predecir_log(D, F, dat))
        L["dl_promedio"] = 0.5 * (L["idw"] + L["dl_hibrido"])
        L["dl_gbm"] = 0.5 * (L["hibrido"] + L["dl_hibrido"])
        return L

    def log_metodos(M, F, placebo=False):
        return con_redes(original_log(M, F, placebo), M["dl"], F, M["dl_dat"])

    def validar(dat, diseños, log, coefs=None, imp_grupos=None, perm=None):
        """La misma validación de v8_modelo.validar (mismos pliegues, mismo orden y mismos números aleatorios), con
        las redes en vuelo: mientras las redes de un pliegue entrenan, la V8 avanza con ese pliegue y con los
        siguientes (hasta dl.profundidad pliegues a la vez), y cada pliegue se cierra cuando sus redes terminan."""
        import pandas as pd
        filas, pendientes = [], []
        rng = np.random.default_rng(m.SEMILLA)

        def cerrar(x):
            D = dl.recoger(x["pend"])
            Fte, p0 = x["Fte"], x["p0"]
            L = con_redes(x["L"], D, Fte, dat)
            met = m.METODOS + ["idw_placebo"]
            base = pd.DataFrame({"diseño": x["nombre"], "pliegue": x["f"], "sid_idx": Fte["sid_idx"].values,
                                 "t": Fte["t"].values, "obs": np.exp(Fte["y"].values),
                                 "dmin_km": Fte["dmin"].values,
                                 **{mt: np.exp(L[mt]) for mt in met},
                                 "vecina": np.exp(Fte["vecina"].values),
                                 "covariables": np.exp(p0.reindex(pd.MultiIndex.from_arrays(
                                     [Fte["sid_idx"], Fte["t"]])).values) * x["smear0"],
                                 **{f"log_res_{mt}": Fte["y"].values - L[mt] for mt in met}})
            filas.append(base)
            log(f"    {x['nombre']} pliegue {x['f']}: {x['n_train']} estaciones de entrenamiento, {x['n_test']} de "
                f"prueba, {x['n_ftr']} filas, {time.time() - x['t0']:.0f} s")

        try:
            for nombre, folds in diseños.items():
                for f in sorted(set(folds)):
                    test = np.where(folds == f)[0]
                    train = np.where(folds != f)[0]
                    if len(train) < 3 or len(test) == 0:
                        continue
                    t0 = time.time()
                    Ftr = dat.filas_entrenamiento(train, rng, True)
                    n_perm = m.N_PERMUTACIONES if (perm is not None and nombre.startswith("kNNDM")) else 0
                    pend = dl.lanzar(Ftr, dat, train)
                    try:
                        M = original_ajustar(Ftr, dat, train, n_perm)
                        Fte = dat.filas_prueba(test, train, True)
                        L = original_log(M, Fte, placebo=True)
                        if n_perm:
                            perm += [{"diseño": nombre, **x} for x in m.nmae_permutaciones(dat, M, Fte, L, test, train)]
                        if coefs is not None and M["ridge"] is not None:
                            coefs.append({"diseño": nombre, "pliegue": f, "alfa": float(M["ridge"].alfa),
                                          "intercepto": float(M["ridge"].intercept_),
                                          **dict(zip(M["cols_du"], M["ridge"].coef_.astype(float)))})
                        if imp_grupos is not None and nombre == "kNNDM":
                            for g, (a, pct, n) in m.importancia_grupos(M["hibrido_suelo"], Fte).items():
                                imp_grupos.append({"pliegue": f, "grupo": g, "variables": n, "aumento_ecm_log": a,
                                                   "aumento_ecm_pct": pct})
                        Ftr0 = dat.filas_entrenamiento(train, rng, False)
                        m0 = m.ajustar(Ftr0)
                        Fte0 = dat.filas_prueba(test, train, False)
                        p0 = pd.Series(m.predecir(m0, Fte0), index=pd.MultiIndex.from_arrays([Fte0["sid_idx"], Fte0["t"]]))
                    except BaseException:
                        dl._liberar(pend["conj"], pend["futuros"])
                        raise
                    pendientes.append({"pend": pend, "Fte": Fte, "L": L, "p0": p0, "smear0": m0.smear_["_todas"],
                                       "nombre": nombre, "f": f, "n_train": len(train), "n_test": len(test),
                                       "n_ftr": len(Ftr), "t0": t0})
                    del M, Ftr, Ftr0, Fte0
                    while len(pendientes) >= dl.profundidad + 1:
                        cerrar(pendientes.pop(0))
            while pendientes:
                cerrar(pendientes.pop(0))
        finally:
            for x in pendientes:
                dl._liberar(x["pend"]["conj"], x["pend"]["futuros"])
        return pd.concat(filas, ignore_index=True)

    def filas_municipio(self, i):
        X = original_mun(self, i)
        X["_punto"] = i                     # para la corrección por inverso de la distancia; ningún otro método la usa
        return X

    m.ajustar_metodos, m.log_metodos, m.Datos.filas_municipio = ajustar_metodos, log_metodos, filas_municipio
    m.validar = validar
    m.Datos.variables, m.Datos.filas_entrenamiento = variables, filas_entrenamiento


# ═══════════════════════════════════════════════════════════════════
# COMPARACIÓN Y CONTROL
# ═══════════════════════════════════════════════════════════════════
def comparar(m, salida, contaminantes, log):
    """Cada método con red frente a idw, el híbrido de árboles, la referencia sin vecinas (covariables) y la mejor
    alternativa sin red del mismo diseño: diferencia media de NMAE por estación (negativa = la red tiene menor
    error), intervalo del 95 % por remuestreo de grupos espaciales y la diferencia sin el grupo más favorable."""
    import numpy as np
    import pandas as pd
    filas = []
    for c in contaminantes:
        f = os.path.join(salida, f"{c}_validacion_por_estacion.csv")
        if not os.path.exists(f):
            continue
        me = pd.read_csv(f, dtype={"sid": str})
        est = pd.read_csv(os.path.join(salida, f"{c}_estaciones.csv"), dtype={"sid": str})
        grupos_est = pd.Series(m.grupos_espaciales(est), index=est["sid"])
        for ds in me["diseño"].unique():
            x = me[me["diseño"] == ds].pivot_table(index="sid", columns="metodo", values="NMAE_pct")
            sin_red = [k for k in m.METODOS if k not in DL_METODOS and k in x]
            mejor = x[sin_red].mean().idxmin()
            grupos = grupos_est.reindex(x.index).values
            ug = np.unique(grupos)
            rng = np.random.default_rng(m.SEMILLA)
            idx = {g: np.where(grupos == g)[0] for g in ug}
            muestras = [np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]) for _ in range(m.N_REMUESTREOS)]
            for red in DL_METODOS:
                for ref in dict.fromkeys(["idw", "hibrido", "covariables", mejor]):
                    if red not in x or ref not in x:
                        continue
                    dlt = (x[red] - x[ref]).values
                    ok = np.isfinite(dlt)
                    if ok.sum() < 5:
                        continue
                    boot = np.array([np.nanmean(dlt[mm]) for mm in muestras])
                    jk = max(np.nanmean(dlt[grupos != g]) for g in ug) if len(ug) > 1 else np.nan
                    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                    filas.append({"contaminante": c, "diseño": ds, "metodo_red": red, "frente_a": ref,
                                  "frente_a_es_la_mejor_sin_red": ref == mejor, "estaciones": int(ok.sum()),
                                  "grupos_espaciales": len(ug),
                                  "NMAE_red_media": float(np.nanmean(x[red])), "NMAE_ref_media": float(np.nanmean(x[ref])),
                                  "diferencia_media_NMAE": float(np.nanmean(dlt)), "ic95_inf": float(lo), "ic95_sup": float(hi),
                                  "diferencia_peor_sin_un_grupo": float(jk),
                                  "lectura": ("la red tiene menor error" if hi < 0 else
                                              "la red tiene mayor error" if lo > 0 else "sin diferencia distinguible")})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(salida, "COMPARACION_DL.csv"), index=False)
    if len(t):
        x = t[(t["diseño"] == "kNNDM") & t["frente_a_es_la_mejor_sin_red"]]
        for r in x.itertuples():
            log(f"  {r.contaminante} {r.metodo_red} frente a {r.frente_a}: {r.diferencia_media_NMAE:+.2f} puntos de NMAE "
                f"(IC 95 % {r.ic95_inf:+.2f} a {r.ic95_sup:+.2f}): {r.lectura}")
    return t


def reproducibilidad(resumen_v83, salida, log):
    import pandas as pd
    a = pd.read_csv(io.BytesIO(resumen_v83))
    b = pd.read_csv(os.path.join(salida, "VALIDACION_RESUMEN.csv"))
    x = a.merge(b, on=["contaminante", "diseño", "metodo"], suffixes=("_v83", "_ahora"))
    for c in ("R2", "r", "NMAE_pct", "error_medio_pct"):
        x[f"dif_{c}"] = x[f"{c}_ahora"] - x[f"{c}_v83"]
    x.to_csv(os.path.join(salida, "REPRODUCIBILIDAD_V8_3.csv"), index=False)
    d = x[[c for c in x if c.startswith("dif_")]].abs().max()
    log(f"Métodos de la V8.3 recalculados frente a los archivados ({len(x)} combinaciones de contaminante, diseño y "
        f"método): diferencia máxima R² {d['dif_R2']:.4f}, NMAE {d['dif_NMAE_pct']:.3f} puntos, "
        f"error medio {d['dif_error_medio_pct']:.3f} puntos")


def empacar(salida, log):
    """Un solo zip en archive/, con la misma estructura de resultados_v8.zip (carpeta resultados/), verificado."""
    if os.path.exists(ZIP_FINAL):
        viejo = os.path.join(os.path.dirname(ZIP_FINAL), "_to_delete",
                             f"resultados_v8_dl_{datetime.now():%Y%m%d_%H%M}.zip")
        os.makedirs(os.path.dirname(viejo), exist_ok=True)
        shutil.move(ZIP_FINAL, viejo)
        log(f"El zip anterior quedó en {viejo}")
    tmp = os.path.join(TRABAJO, "resultados_v8_dl.zip")
    archivos = sorted(f for f in os.listdir(salida) if os.path.isfile(os.path.join(salida, f)))
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in archivos:
            z.write(os.path.join(salida, f), f"resultados/{f}")
    with zipfile.ZipFile(tmp) as z:
        malo = z.testzip()
        if malo or len(z.namelist()) != len(archivos):
            log(f"ATENCIÓN: el zip no quedó bien ({malo}); los resultados siguen en {salida}")
            return
    shutil.copy2(tmp, ZIP_FINAL + ".tmp")
    if sha256(tmp) != sha256(ZIP_FINAL + ".tmp"):
        log(f"ATENCIÓN: la copia en OneDrive no coincide; los resultados siguen en {salida} y {tmp}")
        return
    os.replace(ZIP_FINAL + ".tmp", ZIP_FINAL)
    os.remove(tmp)
    log(f"Resultados en {ZIP_FINAL} ({os.path.getsize(ZIP_FINAL) / 2**20:.0f} MB, {len(archivos)} archivos, verificado)")


LEEME = """Resultados de la V8.3 con aprendizaje profundo ({version}; corrida del {fecha})
Script: 02_scripts/02_analysis/12_v8_dl.py (copia en src/analysis/). Formato: el de resultados_v8.zip (ver
src/model/LEEME_V8.md), con cinco métodos más en todas las tablas:
{metodos}

Las covariables son las de resultados_v8.zip (misma huella, comprobada antes de correr). Los métodos de la V8.3 se
recalcularon en la misma corrida y con los mismos pliegues; REPRODUCIBILIDAD_V8_3.csv los compara con los
archivados. COMPARACION_DL.csv compara cada método con red con idw, el híbrido de árboles, la referencia sin
vecinas (covariables) y la mejor alternativa sin red del mismo diseño: diferencia media del NMAE por estación
(negativa = la red tiene menor error), intervalo del 95 % por remuestreo de grupos de estaciones y diferencia sin el
grupo más favorable. La regla de elección de la V8 (eleccion_metodo.csv) incluye los métodos con red.
DL_ENTRENAMIENTO.csv: cada red entrenada (épocas, mejor época, pérdidas, tiempo). dl_hiperparametros.json: los de
Gregori en la V7.1, sin reajustar. REGISTRO.txt: el registro completo de la corrida.
"""


# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="V8.3 con la red TCN de Gregori como método adicional")
    ap.add_argument("--cpu", action="store_true", help="no usar la tarjeta gráfica aunque la haya")
    ap.add_argument("--prueba", action="store_true",
                    help="corrida corta (PM2.5, 2023-2024, 3 pliegues, 3 épocas) para comprobar todo y estimar la duración")
    a = ap.parse_args()
    comprobar_paquetes()
    os.makedirs(TRABAJO, exist_ok=True)
    salida = os.path.join(TRABAJO, "resultados_prueba" if a.prueba else "resultados")
    os.makedirs(salida, exist_ok=True)
    reg_tmp = []

    def log(x):
        print(x, flush=True)
        reg_tmp.append(x)

    log(f"{VERSION_DL} | {datetime.now():%Y-%m-%d %H:%M} | {platform.node()} | Python {platform.python_version()}")
    log(preparar_equipo())
    c8, firma_v83, resumen_v83 = extraer_insumos(log)
    log("Hiperparámetros de la red (Gregori, V7.1):")
    hp = leer_hiperparametros(log)
    firma = hashlib.sha256((VERSION_DL + firma_v83 + HUELLA_SCRIPT + json.dumps(hp, sort_keys=True)
                            + str(a.prueba)).encode()).hexdigest()
    archivo_firma = os.path.join(salida, "FIRMA_CORRIDA.txt")
    sigue = os.path.exists(archivo_firma) and open(archivo_firma).read().strip() == firma
    if not sigue and os.listdir(salida):
        viejo = os.path.join(TRABAJO, "_to_delete", f"{os.path.basename(salida)}_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(os.path.dirname(viejo), exist_ok=True)
        shutil.move(salida, viejo)
        os.makedirs(salida)
        log(f"Resultados anteriores (otra versión del script o de los datos) movidos a {viejo}")
    open(archivo_firma, "w").write(firma)
    reg = open(os.path.join(salida, "REGISTRO.txt"), "a" if sigue else "w", encoding="utf-8")
    reg.write("\n".join(reg_tmp) + "\n")

    import re
    patron_pliegue = re.compile(r"^\s+(\S+) pliegue (\d+):")
    dl = None

    def log(x):  # noqa: F811
        print(x, flush=True)
        reg.write(x + "\n")
        reg.flush()
        # la V8 anota cada pliegue al terminarlo: las redes entrenadas hasta ahí son de ese pliegue
        r = patron_pliegue.match(x)
        if r and dl is not None:
            for h in dl.historial:
                h.setdefault("diseño", r.group(1))
                h.setdefault("pliegue", int(r.group(2)))

    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    import v8_modelo as m
    log(f"numpy {np.__version__} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} | torch {torch.__version__}")
    json.dump({"hiperparametros_v7_1": hp, "semillas": SEMILLAS, "epocas_max": EPOCAS_MAX, "paciencia": PACIENCIA,
               "ventanas_por_epoca": MUESTRAS_EPOCA, "ventanas_validacion": MUESTRAS_VALIDACION,
               "fraccion_grupos_validacion": FRACCION_VALIDACION, "recorte": RECORTE},
              open(os.path.join(salida, "dl_hiperparametros.json"), "w"), indent=1)
    m.HILOS = NUCLEOS
    log("Midiendo cuántas redes conviene entrenar a la vez en este equipo (1-2 min):")
    procesos, hilos, disp, rendimiento = afinar(hp["pm25"], log, a.cpu)
    dl = DL(m, hp, os.path.join(TRABAJO, "redes_entrenadas_prueba" if a.prueba else "redes_entrenadas"), log, a.prueba,
            procesos, hilos, disp)
    try:
        correr(a, m, dl, c8, log, salida, sigue, resumen_v83, procesos, hilos, rendimiento)
    finally:
        dl.cerrar()
        reg.close()
    if not a.prueba:
        empacar(salida, print)


def correr(a, m, dl, c8, log, salida, sigue, resumen_v83, procesos, hilos, rendimiento):
    import pandas as pd
    import psutil
    instalar_dl(m, dl)
    obs, est, mun, dinam, fijas, fechas = c8.leer_datos(a.prueba)
    if a.prueba:                                          # lo mismo que hace correr_v8.py --prueba
        m.RADIOS_AUMENTO = [0, 20, 80]
        m.HGB["max_iter"] = 100
    contaminantes = ["pm25"] if a.prueba else ["pm25", "pm10", "no2", "o3"]
    t0 = time.time()
    resumenes = {}
    for p in contaminantes:
        if sigue and os.path.exists(os.path.join(salida, f"{p}_modelo.json")):
            log(f"\n=== {p}: terminado en la corrida anterior; se conserva ===")
            resumenes[p] = pd.read_csv(os.path.join(salida, f"{p}_validacion_resumen.csv"), index_col=[0, 1])
            continue
        tp = time.time()
        r = m.correr_contaminante(p, obs, est, mun, dinam, fijas, fechas, salida, log, a.prueba)
        if r is not None:
            resumenes[p] = r
        for h in dl.historial:
            h.setdefault("diseño", "modelo_final")
        pd.DataFrame(dl.historial).to_csv(os.path.join(salida, f"{p}_dl_entrenamiento.csv"), index=False)
        dl.historial = []
        mem = psutil.virtual_memory()
        log(f"  {p}: {(time.time() - tp) / 3600:.1f} h; memoria libre {mem.available / 2**30:.1f} GB"
            + ("  (MENOS DE 3 GB LIBRES)" if mem.available < 3 * 2**30 else ""))
    filas = []
    for p, r in resumenes.items():
        x = r.reset_index()
        x.insert(0, "contaminante", p)
        filas.append(x)
    pd.concat(filas).to_csv(os.path.join(salida, "VALIDACION_RESUMEN.csv"), index=False)
    hist = pd.concat([pd.read_csv(os.path.join(salida, f"{p}_dl_entrenamiento.csv")) for p in contaminantes
                      if os.path.exists(os.path.join(salida, f"{p}_dl_entrenamiento.csv"))], ignore_index=True)
    primeras = [c for c in ("contaminante", "diseño", "pliegue", "red", "semilla") if c in hist]
    hist = hist[primeras + [c for c in hist if c not in primeras]]
    if "pliegue" in hist:
        hist["pliegue"] = hist["pliegue"].astype("Int64")
    hist.to_csv(os.path.join(salida, "DL_ENTRENAMIENTO.csv"), index=False)
    log("\nMétodos con red frente a la mejor alternativa sin red (validación kNNDM, NMAE por estación):")
    comparar(m, salida, contaminantes, log)
    if not a.prueba:
        reproducibilidad(resumen_v83, salida, log)
    with open(os.path.join(salida, "LEEME.txt"), "w", encoding="utf-8") as f:
        f.write(LEEME.format(version=VERSION_DL, fecha=f"{datetime.now():%Y-%m-%d}",
                             metodos="\n".join(f"  {k}: {v}" for k, v in DESC_DL.items())))
    log(f"\nTERMINADO en {(time.time() - t0) / 3600:.1f} h")
    if a.prueba:
        # costo relativo por ventana de la red de cada contaminante frente a la de PM2.5 (medido con sus
        # hiperparámetros); unas 25 épocas por red; 41 ajustes por contaminante (10 + 10 + ~10 + 10 pliegues y el final)
        rel = {"pm25": 1.0, "pm10": 0.21, "no2": 0.98, "o3": 0.39}
        ventanas_red = 25 * (MUESTRAS_EPOCA + MUESTRAS_VALIDACION / 3)
        horas = 41 * 2 * len(SEMILLAS) * ventanas_red * sum(rel.values()) / rendimiento / 3600
        log(f"Estimación aproximada de la corrida completa en este equipo, con la carga de ahora: {horas:.0f} h de "
            f"entrenamiento de las redes ({procesos} a la vez x {hilos} hilos en {dl.disp}, {rendimiento:.0f} ventanas/s), más unas "
            f"2-3 h de la V8, la preparación de los datos y las predicciones.")
        log(f"Prueba terminada. Resultados de prueba en {salida} (no se copian a OneDrive).")


if __name__ == "__main__":
    main()
