# lstm/artifacts.py
# Model artifact persistence for EdgePulse Phase 3.
#
# Saves / loads:
#   - Model state dictionary (PyTorch)
#   - Architecture configuration
#   - Exact LSTM_FEATURES order
#   - Sequence length and hidden dimension
#   - Imputer and scaler parameters (training-data fitted)
#   - Anomaly threshold and calibration metadata (when available)
#   - Training metadata: seed, epochs, dataset identifiers, framework versions
#
# Security:
#   - Model weights loaded with weights_only=True (PyTorch >= 2.0) where supported.
#   - Metadata stored as plain JSON — no arbitrary pickle loading.
#   - Validates feature order and architecture on load.
#   - Raises clear errors for missing, corrupt, or incompatible artifacts.
#
# Atomic writes:
#   - Writes to a temp file first, then renames to final path, so an interrupted
#     write does not leave a corrupt-but-present artifact.

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Default artifact directory (added to .gitignore)
DEFAULT_ARTIFACT_DIR = "models"

# File names within the artifact directory
_WEIGHTS_FILE   = "model_weights.pt"
_METADATA_FILE  = "metadata.json"


def _artifact_dir(model_dir: Optional[str]) -> str:
    return model_dir or DEFAULT_ARTIFACT_DIR


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically: temp file + rename."""
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_save_tensor(path: str, state_dict: dict) -> None:
    """Save PyTorch state dict atomically."""
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ------------------------------------------------------------------

def save_model(
    model,                    # LSTMAutoencoder
    imputer,                  # FeatureImputer
    scaler,                   # FeatureScaler
    feature_names:   list,
    seq_len:         int,
    model_dir:       Optional[str] = None,
    threshold:       Optional[float] = None,
    threshold_percentile: Optional[float] = None,
    threshold_method: str = "percentile",
    train_seed:      Optional[int] = None,
    train_epochs_run: Optional[int] = None,
    best_val_loss:   Optional[float] = None,
    dataset_ids:     Optional[dict] = None,
    model_version:   str = "1.0",
) -> str:
    """
    Save model weights and all metadata required for inference.

    Returns the artifact directory path.

    Parameters
    ----------
    model           : LSTMAutoencoder instance
    imputer         : fitted FeatureImputer
    scaler          : fitted FeatureScaler
    feature_names   : list of feature names in exact canonical order
    seq_len         : sequence length used for training
    model_dir       : artifact directory (default: models/)
    threshold       : calibrated anomaly threshold (None until Stage D)
    threshold_percentile : percentile used for calibration
    threshold_method : calibration method description
    train_seed      : random seed used during training
    train_epochs_run : number of epochs actually trained
    best_val_loss   : validation loss at checkpoint
    dataset_ids     : dict describing training/val/test splits used
    model_version   : human-readable version string
    """
    artifact_dir = _artifact_dir(model_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    # 1. Save weights atomically
    weights_path = os.path.join(artifact_dir, _WEIGHTS_FILE)
    _atomic_save_tensor(weights_path, model.state_dict())
    logger.info("Saved model weights -> %s", weights_path)

    # 2. Build and save metadata
    metadata = {
        "model_version":        model_version,
        "created_at":           datetime.now(timezone.utc).isoformat(),
        "architecture":         model.config(),
        "feature_names":        list(feature_names),
        "seq_len":              seq_len,
        "imputer":              imputer.to_dict(),
        "scaler":               scaler.to_dict(),
        "threshold":            threshold,
        "threshold_percentile": threshold_percentile,
        "threshold_method":     threshold_method,
        "train_seed":           train_seed,
        "train_epochs_run":     train_epochs_run,
        "best_val_loss":        best_val_loss,
        "dataset_ids":          dataset_ids or {},
        "torch_version":        torch.__version__,
    }

    metadata_path = os.path.join(artifact_dir, _METADATA_FILE)
    _atomic_write_json(metadata_path, metadata)
    logger.info("Saved metadata -> %s", metadata_path)

    return artifact_dir


def load_model(
    model_dir:     Optional[str] = None,
    expected_features: Optional[list] = None,
) -> dict:
    """
    Load model and metadata from artifact directory.

    Returns a dict with keys:
        model       : LSTMAutoencoder (weights loaded, eval mode)
        imputer     : FeatureImputer (fitted from saved params)
        scaler      : FeatureScaler  (fitted from saved params)
        metadata    : full metadata dict
        threshold   : float or None

    Raises
    ------
    FileNotFoundError   if artifact directory or files are missing
    ValueError          if feature order or architecture is incompatible
    RuntimeError        if weights cannot be loaded
    """
    artifact_dir = _artifact_dir(model_dir)

    # Check files exist
    weights_path  = os.path.join(artifact_dir, _WEIGHTS_FILE)
    metadata_path = os.path.join(artifact_dir, _METADATA_FILE)

    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(
            f"Artifact directory not found: {artifact_dir!r}. "
            "Run train.py to generate it."
        )
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path!r}. "
            "The artifact directory may be corrupt."
        )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path!r}. "
            "The artifact directory may be corrupt."
        )

    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate feature order
    saved_features = metadata.get("feature_names", [])
    if expected_features is not None and saved_features != list(expected_features):
        raise ValueError(
            f"Feature order mismatch.\n"
            f"  Saved:    {saved_features}\n"
            f"  Expected: {list(expected_features)}"
        )

    # Reconstruct model
    from lstm.model import LSTMAutoencoder
    arch = metadata["architecture"]
    try:
        model = LSTMAutoencoder(
            n_features=arch["n_features"],
            hidden_dim=arch["hidden_dim"],
            seq_len=arch["seq_len"],
            num_layers=arch.get("num_layers", 1),
        )
    except KeyError as e:
        raise ValueError(f"Incomplete architecture metadata: {e}") from e

    # Load weights safely
    try:
        # weights_only=True is safer (PyTorch >= 2.0)
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch versions don't support weights_only
            state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model weights from {weights_path!r}: {e}"
        ) from e

    model.eval()

    # Reconstruct imputer and scaler
    from lstm.dataset import FeatureImputer, FeatureScaler
    imputer = FeatureImputer.from_dict(metadata["imputer"])
    scaler  = FeatureScaler.from_dict(metadata["scaler"])

    threshold = metadata.get("threshold")
    logger.info(
        "Loaded model v%s from %s (threshold=%s)",
        metadata.get("model_version"), artifact_dir, threshold
    )

    return {
        "model":     model,
        "imputer":   imputer,
        "scaler":    scaler,
        "metadata":  metadata,
        "threshold": threshold,
    }


def artifact_exists(model_dir: Optional[str] = None) -> bool:
    """Return True if a complete artifact set exists at model_dir."""
    d = _artifact_dir(model_dir)
    return (
        os.path.isdir(d)
        and os.path.isfile(os.path.join(d, _WEIGHTS_FILE))
        and os.path.isfile(os.path.join(d, _METADATA_FILE))
    )
