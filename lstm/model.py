# lstm/model.py
# LSTM Autoencoder for EdgePulse Phase 3.
#
# Architecture (encoder-decoder):
#   Input:   (batch, seq_len, n_features=12)
#   Encoder: LSTM(n_features -> hidden_dim)  -> last hidden state (batch, hidden_dim)
#   Decoder: repeat hidden_dim -> (batch, seq_len, hidden_dim)
#              LSTM(hidden_dim -> hidden_dim)
#              Linear(hidden_dim -> n_features)
#   Output:  (batch, seq_len, n_features)  -- same shape as input
#
# Loss: MSE reconstruction error (mean over all elements).
#
# Design constraints:
#   - CPU-only capable (no CUDA required).
#   - Small enough to train on a student laptop in seconds.
#   - No attention, no transformers, no unnecessary complexity.
#   - Validates input shape and rejects non-finite values.

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


class LSTMAutoencoder(nn.Module):
    """
    Small LSTM autoencoder for temporal anomaly detection.

    Parameters
    ----------
    n_features  : int   -- number of input features (must be 12 for EdgePulse)
    hidden_dim  : int   -- LSTM hidden state size (default 32)
    seq_len     : int   -- expected sequence length (stored for validation)
    num_layers  : int   -- stacked LSTM layers for encoder and decoder (default 1)
    dropout     : float -- dropout between stacked LSTM layers (only if num_layers>1)
    """

    def __init__(
        self,
        n_features: int = 12,
        hidden_dim: int = 32,
        seq_len:    int = 30,
        num_layers: int = 1,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.seq_len    = seq_len
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0

        # Encoder: maps sequence to a compact hidden state
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Decoder: reconstructs the sequence from the latent state
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Project decoder output back to feature space
        self.output_layer = nn.Linear(hidden_dim, n_features)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)  float32

        Returns
        -------
        reconstruction : (batch, seq_len, n_features)

        Raises
        ------
        ValueError  if x has wrong number of dimensions or features
        ValueError  if x contains non-finite values
        """
        self._validate_input(x)

        batch_size = x.size(0)

        # --- Encode ---
        # encoder_out : (batch, seq_len, hidden_dim)
        # h_n          : (num_layers, batch, hidden_dim)
        _, (h_n, c_n) = self.encoder(x)

        # Use last layer's hidden state as latent representation
        latent = h_n[-1]   # (batch, hidden_dim)

        # --- Decode ---
        # Repeat latent across seq_len to give decoder a fixed input at each step
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        # (batch, seq_len, hidden_dim)

        decoder_out, _ = self.decoder(decoder_input)
        # decoder_out : (batch, seq_len, hidden_dim)

        # Project to feature space
        reconstruction = self.output_layer(decoder_out)
        # (batch, seq_len, n_features)

        return reconstruction

    # ------------------------------------------------------------------
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample MSE reconstruction error.

        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        errors : (batch,)  -- MSE averaged over seq_len and n_features
        """
        with torch.no_grad():
            reconstruction = self.forward(x)
            # MSE per sample: mean over seq_len and feature dims
            errors = ((reconstruction - x) ** 2).mean(dim=(1, 2))
        return errors

    # ------------------------------------------------------------------
    def config(self) -> dict:
        """Return architecture configuration for serialisation."""
        return {
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "seq_len":    self.seq_len,
            "num_layers": self.num_layers,
        }

    # ------------------------------------------------------------------
    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape (batch, seq_len, n_features), "
                f"got {tuple(x.shape)}"
            )
        if x.size(2) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.size(2)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("Input contains non-finite values (NaN or Inf).")


# ------------------------------------------------------------------

def build_model(
    n_features: int = 12,
    hidden_dim: int = 32,
    seq_len:    int = 30,
    num_layers: int = 1,
    seed:       Optional[int] = None,
) -> LSTMAutoencoder:
    """
    Construct and return a new LSTMAutoencoder.
    Sets the torch random seed for deterministic weight initialisation when
    seed is provided.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        num_layers=num_layers,
    )
    return model


def mse_loss(reconstruction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss over all elements (batch mean).
    Both tensors must have shape (batch, seq_len, n_features).
    """
    return nn.functional.mse_loss(reconstruction, original)
