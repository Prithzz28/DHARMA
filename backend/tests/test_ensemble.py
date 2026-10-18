# -*- coding: utf-8 -*-
"""
tests/test_ensemble.py
======================
Unit and integration tests for the Dual-Model ML Ensemble
(RandomForestClassifier + DeFiGuardPredictor).
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import joblib

from inference_engine.services.ensemble import EnsembleMode, ensemble_predict
from inference_engine.services.prediction import run_inference
from inference_engine.api.dependencies import _import_defiguard_stubs, AppState


@pytest.mark.asyncio
async def test_ensemble_max_mode_prefers_higher_score():
    """Verify that MAX mode picks the best (most vigilant) score."""
    mock_rf = MagicMock()
    mock_rf.predict_proba.return_value = [[0.6, 0.4]]  # rf score = 0.40

    mock_dg = MagicMock()
    mock_dg.predict_tabular_proba.return_value = 0.85  # dg score = 0.85

    features = {"length_transaction_hash": 66.0, "block_number": 19500000.0}

    final_score, rf_score, dg_score, mode = await ensemble_predict(
        rf_model=mock_rf,
        dg_model=mock_dg,
        features=features,
        mode=EnsembleMode.MAX,
    )

    assert rf_score == 0.4
    assert dg_score == 0.85
    assert final_score == 0.85
    assert mode == "max"


@pytest.mark.asyncio
async def test_ensemble_max_mode_prefers_rf_when_higher():
    """Verify MAX mode selects RF if RF detected higher risk."""
    mock_rf = MagicMock()
    mock_rf.predict_proba.return_value = [[0.1, 0.9]]  # rf score = 0.90

    mock_dg = MagicMock()
    mock_dg.predict_tabular_proba.return_value = 0.30  # dg score = 0.30

    features = {"length_transaction_hash": 66.0}

    final_score, rf_score, dg_score, mode = await ensemble_predict(
        rf_model=mock_rf,
        dg_model=mock_dg,
        features=features,
        mode=EnsembleMode.MAX,
    )

    assert rf_score == 0.90
    assert dg_score == 0.30
    assert final_score == 0.90
    assert mode == "max"


@pytest.mark.asyncio
async def test_ensemble_fallback_when_dg_none():
    """Verify graceful fallback to RF-only when dg_model is None."""
    mock_rf = MagicMock()
    mock_rf.predict_proba.return_value = [[0.3, 0.7]]

    features = {"length_transaction_hash": 66.0}

    final_score, rf_score, dg_score, mode = await ensemble_predict(
        rf_model=mock_rf,
        dg_model=None,
        features=features,
        mode=EnsembleMode.MAX,
    )

    assert rf_score == 0.7
    assert dg_score is None
    assert final_score == 0.7
    assert mode == "rf_only"


@pytest.mark.asyncio
async def test_run_inference_primary_model_attribution():
    """Verify PredictionResponse reports the correct decisive model."""
    mock_rf = MagicMock()
    mock_rf.predict_proba.return_value = [[0.75, 0.25]]

    mock_dg = MagicMock()
    mock_dg.predict_tabular_proba.return_value = 0.92

    resp = await run_inference(
        model=mock_rf,
        features={"length_transaction_hash": 66.0},
        defiguard_model=mock_dg,
    )

    assert resp.is_fraud is True  # 0.92 > threshold
    assert resp.fraud_probability == 0.92
    assert resp.rf_score == 0.25
    assert resp.defiguard_score == 0.92
    assert resp.primary_model == "defiguard"


def test_real_joblib_models_loading_and_prediction():
    """Integration test: Load both real .joblib files and execute dual inference."""
    _import_defiguard_stubs()

    backend_dir = Path(__file__).resolve().parent.parent
    rf_path = backend_dir / "models" / "random_forest_fraud_model.joblib"
    dg_path = backend_dir / "models" / "defi_guard_model.joblib"

    if not rf_path.exists() or not dg_path.exists():
        pytest.skip("Model files not found in models/")

    state = AppState()
    state.load(rf_path)
    state.load_defiguard(dg_path)

    assert state.is_loaded is True
    assert state.defiguard_loaded is True

    # Run inference on both models
    test_features = {
        "length_transaction_hash": 66.0,
        "length_to": 42.0,
        "length_from": 42.0,
        "block_number": 19500000.0,
        "gas_efficiency": 0.85,
        "chain_id": 1.0,
        "effective_gas_price": 20000000000.0,
        "is_same_address": 0.0,
        "gas_used": 21000.0,
        "value": 1.0e18,
    }

    from inference_engine.services.ensemble import _sync_rf_predict, _sync_dg_predict
    rf_score = _sync_rf_predict(state.model, test_features)
    dg_score = _sync_dg_predict(state.defiguard_model, test_features)

    assert 0.0 <= rf_score <= 1.0
    assert 0.0 <= dg_score <= 1.0
    best_score = max(rf_score, dg_score)
    assert best_score >= rf_score
    assert best_score >= dg_score
