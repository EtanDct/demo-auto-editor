"""Détection matérielle et sélection de profil (plan-technique.md, section 2 et 10).

Détecte la présence d'un GPU CUDA compatible et la VRAM disponible, avec
repli sur CPU si aucun GPU n'est trouvé. Le résultat sert à ajuster
automatiquement la taille du modèle Whisper et le déchargement GPU du LLM.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass

import psutil

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    ram_gb: float
    gpu_available: bool
    vram_gb: float
    profile: str  # "cpu-16gb" | "gpu-8vram" | "gpu-12vram-plus"
    whisper_model_size: str
    llm_n_gpu_layers: int  # -1 = décharger toutes les couches possibles


def _detect_vram_gb() -> float:
    """Interroge nvidia-smi si disponible ; retourne 0 sinon (pas de GPU NVIDIA détecté)."""
    if shutil.which("nvidia-smi") is None:
        return 0.0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        return round(int(first_line) / 1024, 1)
    except (subprocess.SubprocessError, ValueError, IndexError):
        logger.warning("nvidia-smi présent mais lecture VRAM impossible, repli sur CPU.")
        return 0.0


def detect_hardware() -> HardwareProfile:
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    vram_gb = _detect_vram_gb()
    gpu_available = vram_gb > 0

    if not gpu_available:
        profile, whisper_size, n_gpu_layers = "cpu-16gb", "small", 0
    elif vram_gb < 12:
        profile, whisper_size, n_gpu_layers = "gpu-8vram", "small", -1
    else:
        profile, whisper_size, n_gpu_layers = "gpu-12vram-plus", "medium", -1

    return HardwareProfile(
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        vram_gb=vram_gb,
        profile=profile,
        whisper_model_size=whisper_size,
        llm_n_gpu_layers=n_gpu_layers,
    )


def log_hardware_profile(profile: HardwareProfile, logs_dir) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "hardware.json"
    out_path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    logger.info("Profil matériel détecté : %s (log: %s)", profile.profile, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detected = detect_hardware()
    print(json.dumps(asdict(detected), indent=2))
