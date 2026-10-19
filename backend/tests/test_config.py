# -*- coding: utf-8 -*-
"""Settings: .env discovery independent of the working directory, and validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_engine.core import config
from inference_engine.core.config import BACKEND_DIR, REPO_ROOT, Settings


def test_env_files_are_absolute_and_include_repo_root():
    env_files = Settings.model_config["env_file"]
    assert REPO_ROOT / ".env" in env_files
    assert all(Path(p).is_absolute() for p in env_files)


def test_env_file_loaded_regardless_of_cwd(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FRAUD_THRESHOLD=0.42\nRATE_LIMIT_PER_MINUTE=7\n")
    monkeypatch.chdir(tmp_path.parent)  # a directory with no .env of its own
    monkeypatch.delenv("FRAUD_THRESHOLD", raising=False)
    monkeypatch.delenv("RATE_LIMIT_PER_MINUTE", raising=False)
    s = Settings(_env_file=env)
    assert s.fraud_threshold == 0.42
    assert s.rate_limit_per_minute == 7


def test_relative_model_path_resolves_against_backend(monkeypatch):
    monkeypatch.chdir("/")
    s = Settings(_env_file=None, defiguard_model_path=Path("models/x.joblib"))
    assert s.defiguard_model_path == BACKEND_DIR / "models" / "x.joblib"


def test_obsolete_ensemble_settings_ignored_and_reported(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ENSEMBLE_MODE=max\nmodel_path=models/rf.joblib\nFRAUD_THRESHOLD=0.5\n")
    monkeypatch.setattr(config, "ENV_FILES", (env,))
    monkeypatch.setenv("RF_WEIGHT", "0.5")
    for name in ("ENSEMBLE_MODE", "MODEL_PATH", "DG_WEIGHT"):
        monkeypatch.delenv(name, raising=False)

    s = Settings(_env_file=env)
    assert s.fraud_threshold == 0.5
    assert not hasattr(s, "ensemble_mode")
    assert config.obsolete_settings_in_use() == ["ENSEMBLE_MODE", "MODEL_PATH", "RF_WEIGHT"]


def test_no_obsolete_settings_reported_for_clean_config(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FRAUD_THRESHOLD=0.8\n")
    monkeypatch.setattr(config, "ENV_FILES", (env,))
    for name in config.OBSOLETE_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    assert config.obsolete_settings_in_use() == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fraud_threshold": 0.0},
        {"fraud_threshold": 1.0},
        {"rate_limit_per_minute": -1},
        {"llm_max_calls_per_minute": -5},
    ],
    ids=str,
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)
