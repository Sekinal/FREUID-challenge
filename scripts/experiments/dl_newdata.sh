#!/bin/bash
cd /root/freuid
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p data/aux/idspace data/aux/midv
echo "[$(date)] === listando IDSpace (paises disponibles) ==="
python3 -c "
from huggingface_hub import HfApi
try:
    info=HfApi().repo_info(\"cactuslab/IDSpace\", repo_type=\"dataset\", files_metadata=True)
    sibs=sorted([(s.rfilename,(s.size or 0)/1e9) for s in info.siblings], key=lambda x:x[1])
    for n,gb in sibs[:25]: print(f\"{gb:6.2f}GB {n}\")
except Exception as e: print(\"IDSpace err:\", str(e)[:100])
"
echo "[$(date)] === listando MIDV opciones en HF ==="
python3 -c "
from huggingface_hub import HfApi
for repo in [\"Noaman/midv500\",\"ruturajnawale/midv500\"]:
    try:
        info=HfApi().repo_info(repo, repo_type=\"dataset\", files_metadata=True)
        tot=sum((s.size or 0) for s in info.siblings)/1e9
        print(f\"{repo}: {len(info.siblings)} files, {tot:.1f}GB\")
    except Exception as e: print(repo,\"err\",str(e)[:60])
"
echo "[$(date)] LISTING DONE"
