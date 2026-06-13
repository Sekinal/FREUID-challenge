# Servidor GPU del equipo

Entorno principal para EDA pesado, embeddings y entrenamiento futuro.

## Conexión

```bash
ssh root@216.81.248.172 -p 40299
```

| Campo | Valor |
|-------|--------|
| Host | `216.81.248.172` |
| Puerto | `40299` |
| Usuario | `root` |
| GPU | NVIDIA A100-SXM4-80GB (~80 GB VRAM) |
| Stack | Python 3.12, venv en `/root/freuid/.venv` |

## Rutas en el servidor

- **Proyecto FREUID:** `/root/freuid`
- **Datos Kaggle:** `/root/freuid/data/extracted/` (~17 GB extraído)
- **ZIP descargado:** `/root/freuid/data/raw/the-freuid-challenge-2026-ijcai-ecai.zip` (~16.3 GB)

## Estado actual (2026-06-13)

- Repo sincronizado en `/root/freuid`
- Token Kaggle en `~/.kaggle/access_token`
- Dataset completo descargado y extraído:
  - `train/train/` — 69,352 imágenes (~15 GB)
  - `public_test/` — 7,821 imágenes (~1.7 GB)
  - `train_labels.csv`, `sample_submission.csv`

## Primer uso en el servidor

```bash
ssh root@216.81.248.172 -p 40299
cd /root/freuid

# Dependencias (venv ya creado; recrear si hace falta)
python3 -m venv .venv
.venv/bin/pip install 'kaggle>=2.2.0'
# o, si uv está instalado:
# uv sync

# Kaggle: token SOLO en el servidor (no subir por git)
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
nano ~/.kaggle/access_token   # pegar una línea KGAT_...
chmod 600 ~/.kaggle/access_token

.venv/bin/python scripts/00_download.py
```

## Jobs largos

Usar `tmux` o `screen` para que no se corten al cerrar SSH:

```bash
tmux new -s freuid
bash scripts/run_eda.sh
# Ctrl+B, D para detach
```

## Sincronización Mac ↔ servidor

| Qué | Cómo |
|-----|------|
| Código | `git push` desde Mac → `git pull` en servidor (o `rsync` si el repo es privado) |
| Datos (`data/`, `embeddings/`) | Viven en el servidor; no van en git |
| Credenciales | `LOCAL_CREDENTIALS.txt` **solo en Mac**; en servidor solo `~/.kaggle/access_token` |

Copiar token al servidor (desde tu Mac, si hace falta):

```bash
scp -P 40299 ~/.kaggle/access_token root@216.81.248.172:~/.kaggle/access_token
```

## Auxiliary datasets

Download on the server under `data/aux/` (see [`AUX_DATASETS.md`](AUX_DATASETS.md)).
Do not fill the Mac disk with IDNet-sized archives.

## Coordinación (2 personas)

Avisar en el chat del equipo quién está entrenando o usando la GPU antes de lanzar jobs largos.

## Servidor anterior

`154.54.100.40:40298` (RTX PRO 6000) — ya no usar.
