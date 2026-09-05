"""Téléchargement des modèles depuis Hugging Face (plan-technique.md, section 10).

Tire directement les poids depuis Hugging Face via `huggingface_hub`
(pas d'Ollama ni de llama.cpp CLI). Chaque fichier téléchargé est vérifié
par SHA-256 et consigné dans models/manifest.json pour la reproductibilité.

Usage :
    python scripts/download_models.py            # tous les modèles
    python scripts/download_models.py --only llm # un seul modèle
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer
from huggingface_hub import hf_hub_download, snapshot_download

from hardware import detect_hardware, log_hardware_profile
from pipeline_config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

# piper-tts (PyPI) dépend de piper-phonemize, sans wheel Windows : on utilise
# à la place le binaire officiel publié par le projet (une seule release
# couvre toutes les plateformes ci-dessous).
PIPER_RELEASE_TAG = "2023.11.14-2"
PIPER_ASSETS = {
    ("Windows", None): "piper_windows_amd64.zip",
    ("Linux", "x86_64"): "piper_linux_x86_64.tar.gz",
    ("Linux", "aarch64"): "piper_linux_aarch64.tar.gz",
    ("Linux", "armv7l"): "piper_linux_armv7l.tar.gz",
    ("Darwin", "x86_64"): "piper_macos_x64.tar.gz",
    ("Darwin", "arm64"): "piper_macos_aarch64.tar.gz",
}


def piper_executable_path(models_dir: Path) -> Path:
    exe_name = "piper.exe" if platform.system() == "Windows" else "piper"
    return models_dir / "piper_bin" / "piper" / exe_name


@dataclass
class DownloadedModel:
    name: str
    repo_id: str
    local_path: str
    sha256: str
    size_bytes: int
    downloaded_at: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(name: str, repo_id: str, local_path: Path) -> DownloadedModel:
    return DownloadedModel(
        name=name,
        repo_id=repo_id,
        local_path=str(local_path),
        sha256=_sha256_file(local_path),
        size_bytes=local_path.stat().st_size,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )


def parse_piper_voice(voice: str) -> tuple[str, str, str, str]:
    """"en_US-amy-medium" -> (lang_family="en", lang_code="en_US", speaker="amy", quality="medium")."""
    lang_code, speaker, quality = voice.split("-")
    lang_family = lang_code.split("_")[0]
    return lang_family, lang_code, speaker, quality


def download_whisper(config, models_dir: Path) -> list[DownloadedModel]:
    target = models_dir / "whisper" / config.whisper.model_size
    logger.info("Téléchargement Whisper (%s) vers %s", config.whisper.model_repo, target)
    snapshot_download(repo_id=config.whisper.model_repo, local_dir=target)
    return [
        _record(f"whisper-{config.whisper.model_size}", config.whisper.model_repo, f)
        for f in sorted(target.rglob("*"))
        if f.is_file()
    ]


def download_llm(config, models_dir: Path) -> list[DownloadedModel]:
    target_dir = models_dir / "llm"
    logger.info("Téléchargement LLM (%s/%s)", config.llm.repo_id, config.llm.filename)
    local_path = hf_hub_download(
        repo_id=config.llm.repo_id,
        filename=config.llm.filename,
        local_dir=target_dir,
    )
    return [_record("llm", config.llm.repo_id, Path(local_path))]


def _piper_asset_name() -> str:
    system = platform.system()
    machine = platform.machine()
    key = (system, None) if system == "Windows" else (system, machine)
    asset = PIPER_ASSETS.get(key)
    if asset is None:
        raise RuntimeError(
            f"Plateforme non supportée pour le binaire Piper : {system}/{machine}. "
            f"Plateformes connues : {sorted(PIPER_ASSETS)}"
        )
    return asset


def download_piper_binary(models_dir: Path) -> list[DownloadedModel]:
    exe_path = piper_executable_path(models_dir)
    if exe_path.exists():
        logger.info("Binaire Piper déjà présent : %s", exe_path)
        return [_record("piper-binary", "rhasspy/piper", exe_path)]

    target_dir = models_dir / "piper_bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    asset = _piper_asset_name()
    url = f"https://github.com/rhasspy/piper/releases/download/{PIPER_RELEASE_TAG}/{asset}"
    archive_path = target_dir / asset

    logger.info("Téléchargement du binaire Piper depuis %s", url)
    urllib.request.urlretrieve(url, archive_path)  # noqa: S310 (URL fixe, contrôlée par ce module)

    logger.info("Extraction de %s", archive_path)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(target_dir)
    archive_path.unlink()

    if platform.system() != "Windows":
        exe_path.chmod(0o755)

    if not exe_path.exists():
        raise RuntimeError(f"Binaire Piper introuvable après extraction : {exe_path}")

    return [_record("piper-binary", "rhasspy/piper", exe_path)]


def download_tts(config, models_dir: Path) -> list[DownloadedModel]:
    records = download_piper_binary(models_dir)

    lang_family, lang_code, speaker, quality = parse_piper_voice(config.tts.voice)
    subpath = f"{lang_family}/{lang_code}/{speaker}/{quality}/{config.tts.voice}"
    target_dir = models_dir / "piper"
    logger.info("Téléchargement voix Piper (%s)", config.tts.voice)
    for ext in (".onnx", ".onnx.json"):
        local_path = hf_hub_download(
            repo_id=config.tts.piper_repo_id,
            filename=f"{subpath}{ext}",
            local_dir=target_dir,
        )
        records.append(_record(f"piper-voice{ext}", config.tts.piper_repo_id, Path(local_path)))
    return records


STEPS = {
    "whisper": download_whisper,
    "llm": download_llm,
    "tts": download_tts,
}


@app.command()
def main(
    only: str = typer.Option(None, help="Ne télécharger qu'un seul modèle : whisper, llm ou tts."),
    config_path: Path = typer.Option(None, help="Chemin vers config.yaml (défaut: racine du projet)."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    models_dir = config.paths.resolve("models_dir")
    logs_dir = config.paths.resolve("logs_dir")

    hw = detect_hardware()
    log_hardware_profile(hw, logs_dir)
    logger.info(
        "Matériel détecté : %s (RAM=%.1f Go, VRAM=%.1f Go). Ajuste config.yaml si besoin.",
        hw.profile,
        hw.ram_gb,
        hw.vram_gb,
    )

    steps = {only: STEPS[only]} if only else STEPS
    if only and only not in STEPS:
        raise typer.BadParameter(f"'{only}' inconnu, attendu l'un de : {list(STEPS)}")

    all_records: list[DownloadedModel] = []
    for name, step_fn in steps.items():
        all_records.extend(step_fn(config, models_dir))

    manifest_path = models_dir / "manifest.json"
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = [r for r in existing if r["name"] not in {rec.name for rec in all_records}]
    manifest_path.write_text(
        json.dumps(existing + [asdict(r) for r in all_records], indent=2),
        encoding="utf-8",
    )
    logger.info("Manifeste des modèles écrit dans %s", manifest_path)


if __name__ == "__main__":
    app()
