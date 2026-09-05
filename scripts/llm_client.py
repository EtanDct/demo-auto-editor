"""Chargement du LLM local (llama-cpp-python) pour la traduction (étape C).

Le modèle est un fichier GGUF téléchargé depuis Hugging Face par
scripts/download_models.py (voir config.yaml: llm.repo_id / llm.filename).
"""

from __future__ import annotations

import logging
from pathlib import Path

from hardware import detect_hardware
from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def _resolve_n_gpu_layers(config: PipelineConfig) -> int:
    if config.llm.n_gpu_layers == "auto":
        return detect_hardware().llm_n_gpu_layers
    return int(config.llm.n_gpu_layers)


def load_llm(config: PipelineConfig):
    from llama_cpp import Llama  # import tardif : coûteux, inutile pour --help

    model_path = config.paths.resolve("models_dir") / "llm" / config.llm.filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle LLM introuvable : {model_path}. Lance d'abord "
            "'python scripts/download_models.py --only llm'."
        )

    n_gpu_layers = _resolve_n_gpu_layers(config)
    logger.info("Chargement LLM %s (n_gpu_layers=%d)", model_path.name, n_gpu_layers)
    return Llama(
        model_path=str(model_path),
        n_ctx=config.llm.n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
