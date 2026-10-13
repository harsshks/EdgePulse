# lstm/train.py
# Explicit offline training script for the EdgePulse Phase 3 LSTM autoencoder.
#
# Usage (module):
#   from lstm.train import train
#   result = train(train_split, val_split, ...)
#
# Usage (CLI):
#   python -m lstm.train [--seq-len N] [--hidden-dim N] [--epochs N]
#                        [--batch-size N] [--lr F] [--seed N]
#                        [--model-dir PATH] [--train-steps N] [--val-steps N]
#
# Training policy:
#   - Trains ONLY on healthy training sequences.
#   - Validates on healthy validation sequences.
#   - Never uses test data.
#   - Early stopping: restore best checkpoint when val loss stops improving.
#   - Saves best model via lstm/artifacts.py after training.
#
# Threshold calibration is NOT done here.
# It belongs in lstm/inference.py (Stage D) using val reconstruction errors.

from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lstm.model import LSTMAutoencoder, build_model, mse_loss
from lstm.artifacts import save_model
from lstm.dataset import DatasetSplit, LSTM_FEATURES

logger = logging.getLogger(__name__)

# Default hyperparameters
DEFAULT_SEQ_LEN    = 30
DEFAULT_HIDDEN_DIM = 32
DEFAULT_EPOCHS     = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 1e-3
DEFAULT_PATIENCE   = 10
DEFAULT_SEED       = 42


@dataclass
class TrainResult:
    """Summary of a completed training run."""
    epochs_run:        int
    best_val_loss:     float
    train_losses:      list      # per-epoch training loss
    val_losses:        list      # per-epoch validation loss
    stopped_early:     bool
    artifact_dir:      str
    model:             LSTMAutoencoder


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sequences_to_tensor(sequences: list) -> torch.Tensor:
    """Stack LSTMSequence.features into a float32 tensor (N, seq_len, n_features)."""
    if not sequences:
        return torch.empty(0)
    arrays = [s.features for s in sequences]
    return torch.from_numpy(np.stack(arrays)).float()


def train(
    train_split:   DatasetSplit,
    val_split:     DatasetSplit,
    seq_len:       int   = DEFAULT_SEQ_LEN,
    hidden_dim:    int   = DEFAULT_HIDDEN_DIM,
    max_epochs:    int   = DEFAULT_EPOCHS,
    batch_size:    int   = DEFAULT_BATCH_SIZE,
    lr:            float = DEFAULT_LR,
    patience:      int   = DEFAULT_PATIENCE,
    seed:          int   = DEFAULT_SEED,
    model_dir:     Optional[str] = None,
    model_version: str   = "1.0",
    dataset_ids:   Optional[dict] = None,
) -> TrainResult:
    """
    Train the LSTM autoencoder and save the best checkpoint.

    Parameters
    ----------
    train_split : DatasetSplit with split='train' (healthy sequences only)
    val_split   : DatasetSplit with split='val'   (healthy sequences only)
    seq_len     : sequence length (must match splits)
    hidden_dim  : LSTM hidden state size
    max_epochs  : upper bound on training epochs
    batch_size  : mini-batch size
    lr          : Adam learning rate
    patience    : early stopping patience (epochs without val improvement)
    seed        : RNG seed for reproducibility
    model_dir   : artifact output directory
    model_version : string version tag for the saved artifact
    dataset_ids : optional dict describing which simulation runs were used

    Returns
    -------
    TrainResult with training history and saved artifact path.

    Raises
    ------
    ValueError  if splits are empty or contain non-healthy sequences in training
    """
    # Validate inputs
    if not train_split.sequences:
        raise ValueError("train_split is empty. Cannot train on zero sequences.")
    if not val_split.sequences:
        raise ValueError("val_split is empty. Cannot validate on zero sequences.")
    if train_split.n_anomalous > 0:
        raise ValueError(
            f"train_split contains {train_split.n_anomalous} anomalous sequences. "
            "Training must use healthy sequences only."
        )
    if val_split.n_anomalous > 0:
        raise ValueError(
            f"val_split contains {val_split.n_anomalous} anomalous sequences. "
            "Validation must use healthy sequences only."
        )

    n_features = len(train_split.feature_names)
    actual_seq_len = train_split.sequences[0].features.shape[0]
    if actual_seq_len != seq_len:
        logger.warning(
            "seq_len argument %d differs from actual sequence length %d. "
            "Using actual: %d", seq_len, actual_seq_len, actual_seq_len
        )
        seq_len = actual_seq_len

    _set_seed(seed)

    # Build tensors
    X_train = _sequences_to_tensor(train_split.sequences)  # (N, seq_len, n_features)
    X_val   = _sequences_to_tensor(val_split.sequences)

    logger.info(
        "Training | n_train=%d n_val=%d seq_len=%d n_features=%d hidden=%d",
        len(X_train), len(X_val), seq_len, n_features, hidden_dim,
    )

    # DataLoader (training only — validation computed in full batches)
    train_dataset = TensorDataset(X_train)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = build_model(
        n_features=n_features,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        seed=seed,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_val_loss   = float("inf")
    best_state_dict = None
    patience_count  = 0
    train_losses:   list = []
    val_losses:     list = []
    stopped_early   = False

    for epoch in range(1, max_epochs + 1):
        # --- Training ---
        model.train()
        epoch_train_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            reconstruction = model(batch_x)
            loss = mse_loss(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(batch_x)
        epoch_train_loss /= len(X_train)
        train_losses.append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_reconstruction = model(X_val)
            epoch_val_loss = mse_loss(val_reconstruction, X_val).item()
        val_losses.append(epoch_val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch, max_epochs, epoch_train_loss, epoch_val_loss,
            )

        # --- Early stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss   = epoch_val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info("Early stopping at epoch %d (patience=%d).", epoch, patience)
                stopped_early = True
                break

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best model (val_loss=%.6f).", best_val_loss)

    model.eval()

    # Save artifact
    artifact_dir = save_model(
        model=model,
        imputer=train_split.imputer,
        scaler=train_split.scaler,
        feature_names=train_split.feature_names,
        seq_len=seq_len,
        model_dir=model_dir,
        train_seed=seed,
        train_epochs_run=len(train_losses),
        best_val_loss=best_val_loss,
        dataset_ids=dataset_ids,
        model_version=model_version,
    )

    logger.info("Training complete. Artifact saved to: %s", artifact_dir)

    return TrainResult(
        epochs_run=len(train_losses),
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        stopped_early=stopped_early,
        artifact_dir=artifact_dir,
        model=model,
    )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EdgePulse Phase 3 LSTM Autoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m lstm.train
  python -m lstm.train --seq-len 30 --hidden-dim 32 --epochs 100 --seed 42
  python -m lstm.train --model-dir models/v2 --train-steps 1200 --val-steps 600
        """,
    )
    parser.add_argument("--seq-len",     type=int,   default=DEFAULT_SEQ_LEN)
    parser.add_argument("--hidden-dim",  type=int,   default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--patience",    type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--model-dir",   type=str,   default=None)
    parser.add_argument("--train-steps", type=int,   default=600)
    parser.add_argument("--val-steps",   type=int,   default=600)
    parser.add_argument("--test-steps",  type=int,   default=900)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _parse_args()

    logger.info("Preparing dataset (train_steps=%d, val_steps=%d) ...",
                args.train_steps, args.val_steps)

    from lstm.dataset import prepare_standard_splits
    train_split, val_split, _ = prepare_standard_splits(
        seq_len=args.seq_len,
        train_steps=args.train_steps,
        val_steps=args.val_steps,
        test_steps=args.test_steps,
    )

    logger.info(
        "Dataset ready | train=%d seq | val=%d seq",
        len(train_split.sequences), len(val_split.sequences),
    )

    result = train(
        train_split=train_split,
        val_split=val_split,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        model_dir=args.model_dir,
        dataset_ids={"train_seed": 42, "val_seed": 43, "test_seed": 44},
    )

    print(f"\nTraining complete.")
    print(f"  Epochs run:    {result.epochs_run}")
    print(f"  Best val loss: {result.best_val_loss:.6f}")
    print(f"  Early stopped: {result.stopped_early}")
    print(f"  Artifact dir:  {result.artifact_dir}")


if __name__ == "__main__":
    main()
