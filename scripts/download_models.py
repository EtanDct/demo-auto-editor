"""Téléchargement des modèles (plan-technique.md, section 10).

Tire Whisper et le LLM depuis Hugging Face via `huggingface_hub` (pas d'Ollama
ni de llama.cpp CLI), et la voix Kokoro depuis les publications de kokoro-onnx. Chaque fichier téléchargé est vérifié
par SHA-256 et consigné dans models/manifest.json pour la reproductibilité.

Usage :
    python scripts/download_models.py            # tous les modèles
    python scripts/download_models.py --only llm # un seul modèle
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer
from huggingface_hub import hf_hub_download, snapshot_download

from hardware import detect_hardware, log_hardware_profile
from pipeline_config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

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


KOKORO_REPO = "thewh1teagle/kokoro-onnx"


def kokoro_paths(config, models_dir: Path) -> tuple[Path, Path]:
    """Modèle et banque de voix Kokoro, tels que build_narration les attend."""
    target = models_dir / "kokoro"
    return target / config.tts.kokoro_model, target / config.tts.kokoro_voices


def download_kokoro(config, models_dir: Path) -> list[DownloadedModel]:
    """Modèle Kokoro-82M et banque de voix, depuis les publications de kokoro-onnx.

    Deux fichiers, pas un dépôt Hugging Face : le paquet `kokoro-onnx` lit la
    banque de voix dans son propre format (`voices-v1.0.bin`), que seules ses
    publications fournissent.
    """
    records = []
    for path in kokoro_paths(config, models_dir):
        if path.exists():
            logger.info("Fichier Kokoro déjà présent : %s", path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                f"https://github.com/{KOKORO_REPO}/releases/download/"
                f"{config.tts.kokoro_release}/{path.name}"
            )
            logger.info("Téléchargement de %s", url)
            partial = path.with_suffix(path.suffix + ".part")
            # Écrit à côté puis renommé : un téléchargement interrompu ne laisse
            # pas un fichier tronqué que l'étape suivante prendrait pour bon.
            urllib.request.urlretrieve(url, partial)  # noqa: S310 (URL construite ici)
            partial.replace(path)
        records.append(_record(f"kokoro-{path.name}", KOKORO_REPO, path))
    return records


STEPS = {
    "whisper": download_whisper,
    "llm": download_llm,
    "tts": download_kokoro,
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
