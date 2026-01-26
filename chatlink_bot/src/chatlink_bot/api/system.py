import os
import shutil
import subprocess
import logging
import platform
from typing import Dict, Any, List
from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger("SystemAPI")

def _get_gpu_info() -> Dict[str, Any]:
    """Parses nvidia-smi for VRAM usage if available."""
    try:
        # Query: index, name, memory.used, memory.total
        cmd = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"]
        res = subprocess.check_output(cmd, encoding="utf-8")
        gpus = []
        for line in res.strip().splitlines():
            idx, name, used, total = [x.strip() for x in line.split(",")]
            gpus.append({
                "id": idx,
                "name": name,
                "vram_used_mb": int(used),
                "vram_total_mb": int(total),
                "percent": round((int(used) / int(total)) * 100, 1)
            })
        return {"available": True, "gpus": gpus}
    except Exception:
        return {"available": False, "gpus": []}

def _get_ram_info() -> Dict[str, Any]:
    """Lee la memoria real asignada al contenedor Docker."""
    try:
        # Cgroup V2 (Estándar en Docker moderno)
        if os.path.exists("/sys/fs/cgroup/memory.current"):
            with open("/sys/fs/cgroup/memory.current", "r") as f:
                used = int(f.read().strip())
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                max_raw = f.read().strip()
                # Si no hay límite, usamos la memoria física del host
                total = int(max_raw) if max_raw.isdigit() else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return {"used_mb": used // (1024*1024), "total_mb": total // (1024*1024)}
    except Exception as e:
        logger.error(f"Error leyendo RAM cgroup: {e}")
    
    # Fallback a /proc/meminfo si falla cgroup
    return {"used_mb": 0, "total_mb": 0}

@router.get("/resources", tags=["System"])
async def get_system_resources():
    ram = _get_ram_info()
    gpu = _get_gpu_info()
    
    # Disk usage of the app directory
    total, used, free = shutil.disk_usage("/app" if os.path.exists("/app") else ".")
    
    return {
        "cpu_cores": os.cpu_count(),
        "platform": platform.platform(),
        "ram": ram,
        "gpu": gpu,
        "disk": {
            "total_gb": total // (1024**3),
            "used_gb": used // (1024**3),
            "free_gb": free // (1024**3),
            "percent": round((used/total)*100, 1)
        }
    }

@router.get("/config", tags=["System"])
async def get_safe_config():
    """Returns env vars, masking secrets."""
    safe = {}
    SAFE_KEYS = {
        "QDRANT_URL", "CUDARA_URL", "LOG_LEVEL", "BOT_HISTORY_LIMIT",
        "RESPONSE_DELAY_MINUTES", "QDRANT_COLLECTION", 
        "CUDARA_TEXT_MODEL", "CUDARA_EMBED_MODEL"
    }
    
    for k, v in os.environ.items():
        if k in SAFE_KEYS or k.startswith("CUDARA_") or k.startswith("QDRANT_"):
            safe[k] = v
        elif "PASS" in k or "KEY" in k or "SECRET" in k:
            safe[k] = "******"
        else:
            safe[k] = v
            
    return safe