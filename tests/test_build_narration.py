"""Tests de la voix off (étape D) : choix du moteur, écriture des WAV, fichiers Kokoro.

La synthèse réelle charge un modèle de 325 Mo : elle n'est testée que si les
fichiers sont déjà téléchargés, et le reste ne dépend d'aucun modèle.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

import download_models
from build_narration import KokoroVoice, load_voice, write_wav
from download_models import download_kokoro, kokoro_paths
from pipeline_config import PROJECT_ROOT


def read_wav(path: Path):
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        return wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth(), frames


# --- écriture des WAV -------------------------------------------------------

def test_le_wav_est_mono_16_bits_a_la_frequence_du_moteur(tmp_path):
    out = tmp_path / "seg-001.wav"

    duration = write_wav(np.zeros(24000, dtype=np.float32), 24000, out)

    rate, channels, width, _ = read_wav(out)
    assert (rate, channels, width) == (24000, 1, 2)
    assert duration == 1.0


def test_la_duree_rendue_est_celle_des_echantillons(tmp_path):
    """C'est cette durée que le recalage utilise pour placer chaque phrase."""
    assert write_wav(np.zeros(36000, dtype=np.float32), 24000, tmp_path / "a.wav") == 1.5


def test_les_cretes_sont_ecretees_plutot_que_de_boucler(tmp_path):
    """Sans écrêtage, 1.2 converti en 16 bits déborderait et changerait de signe :
    un claquement au lieu d'une saturation."""
    out = tmp_path / "loud.wav"

    write_wav(np.array([1.2, -1.2, 0.5], dtype=np.float32), 24000, out)

    samples = np.frombuffer(read_wav(out)[3], dtype="<i2")
    assert samples.tolist() == [32767, -32767, 16383]


def test_une_synthese_vide_est_une_erreur_explicite(tmp_path):
    with pytest.raises(ValueError, match="vide"):
        write_wav(np.array([], dtype=np.float32), 24000, tmp_path / "empty.wav")


# --- choix du moteur --------------------------------------------------------

def test_kokoro_af_heart_est_la_voix_par_defaut(config):
    assert (config.tts.engine, config.tts.voice) == ("kokoro", "af_heart")


def test_un_moteur_inconnu_est_refuse_par_la_configuration(config):
    with pytest.raises(ValueError):
        type(config.tts).model_validate({**config.tts.model_dump(), "engine": "espeak"})


def test_sans_fichiers_kokoro_le_message_dit_comment_les_obtenir(config, monkeypatch, tmp_path):
    patched = config.model_copy(deep=True)
    monkeypatch.setattr(type(patched.paths), "resolve", lambda self, field: tmp_path)

    with pytest.raises(FileNotFoundError, match="download_models.py --only tts"):
        load_voice(patched)


def test_sans_binaire_piper_le_message_dit_comment_l_obtenir(config, monkeypatch, tmp_path):
    patched = config.model_copy(deep=True)
    patched.tts.engine = "piper"
    patched.tts.voice = "en_US-amy-medium"
    monkeypatch.setattr(type(patched.paths), "resolve", lambda self, field: tmp_path)

    with pytest.raises(FileNotFoundError, match="Piper"):
        load_voice(patched)


# --- téléchargement ---------------------------------------------------------

def test_les_fichiers_kokoro_suivent_la_configuration(config, tmp_path):
    model, voices = kokoro_paths(config, tmp_path)

    assert model == tmp_path / "kokoro" / "kokoro-v1.0.onnx"
    assert voices == tmp_path / "kokoro" / "voices-v1.0.bin"


def test_le_telechargement_passe_par_un_fichier_partiel(config, monkeypatch, tmp_path):
    """Interrompu, il ne laisse pas un fichier tronqué pris pour bon."""
    seen = []

    def fake_retrieve(url, destination):
        seen.append((url, Path(destination).name))
        Path(destination).write_bytes(b"model")

    monkeypatch.setattr(download_models.urllib.request, "urlretrieve", fake_retrieve)

    records = download_kokoro(config, tmp_path)

    assert [name for _, name in seen] == ["kokoro-v1.0.onnx.part", "voices-v1.0.bin.part"]
    assert seen[0][0].endswith("/model-files-v1.0/kokoro-v1.0.onnx")
    assert all(path.exists() for path in kokoro_paths(config, tmp_path))
    assert not list((tmp_path / "kokoro").glob("*.part"))
    assert [r.name for r in records] == ["kokoro-kokoro-v1.0.onnx", "kokoro-voices-v1.0.bin"]


def test_un_fichier_deja_present_n_est_pas_retelecharge(config, monkeypatch, tmp_path):
    for path in kokoro_paths(config, tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"deja la")

    def fail(*args):
        raise AssertionError("téléchargement inutile")

    monkeypatch.setattr(download_models.urllib.request, "urlretrieve", fail)

    assert len(download_kokoro(config, tmp_path)) == 2


# --- synthèse réelle --------------------------------------------------------

def _kokoro_files_present(config) -> bool:
    return all(p.exists() for p in kokoro_paths(config, PROJECT_ROOT / config.paths.models_dir))


@pytest.fixture
def kokoro(config):
    if not _kokoro_files_present(config):
        pytest.skip("Fichiers Kokoro absents : python scripts/download_models.py --only tts")
    return KokoroVoice(config)


def test_kokoro_synthetise_une_phrase(kokoro, tmp_path):
    out = tmp_path / "seg-001.wav"

    duration = kokoro.synthesize("Click Clear.", out)

    rate, channels, _, frames = read_wav(out)
    assert (rate, channels) == (24000, 1)
    assert 0.3 < duration < 3.0
    assert np.abs(np.frombuffer(frames, dtype="<i2")).max() > 1000  # pas du silence


def test_une_voix_kokoro_inconnue_propose_les_voix_proches(config):
    if not _kokoro_files_present(config):
        pytest.skip("Fichiers Kokoro absents")
    patched = config.model_copy(deep=True)
    patched.tts.voice = "af_hearth"

    with pytest.raises(ValueError, match="af_heart"):
        KokoroVoice(patched)
