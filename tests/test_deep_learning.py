"""Unit tests for Phase 4 deep learning sequence models (LSTM and GRU)."""

from pathlib import Path
import joblib
import numpy as np
import pytest
import torch

from src.models.lstm import (
    EarlyStopping,
    TrafficGRU,
    TrafficLSTM,
    create_sliding_windows,
)


def test_sliding_window_creation():
    """Verify temporal sliding window dimensions and target alignment."""
    features = np.zeros((100, 5), dtype=np.float32)
    # Put sequential values in column 0 (traffic_volume)
    features[:, 0] = np.arange(100, dtype=np.float32)

    X, y = create_sliding_windows(features, seq_len=24, horizon=6)

    # Expected count: 100 - 24 - 6 + 1 = 71
    assert X.shape == (71, 24, 5)
    assert y.shape == (71, 6)

    # First sequence targets should match values 24..29
    np.testing.assert_array_equal(y[0], np.arange(24, 30, dtype=np.float32))


def test_lstm_and_gru_forward_shapes():
    """Verify LSTM and GRU forward pass produces multi-horizon outputs (N, 6)."""
    batch_size = 8
    seq_len = 24
    in_features = 13
    horizon = 6

    x = torch.randn(batch_size, seq_len, in_features)

    lstm = TrafficLSTM(input_size=in_features, hidden_size=32, output_size=horizon)
    gru = TrafficGRU(input_size=in_features, hidden_size=32, output_size=horizon)

    out_lstm = lstm(x)
    out_gru = gru(x)

    assert out_lstm.shape == (batch_size, horizon)
    assert out_gru.shape == (batch_size, horizon)
    assert not torch.isnan(out_lstm).any()
    assert not torch.isnan(out_gru).any()


def test_early_stopping_trigger():
    """Verify early stopping increments counter and triggers appropriately."""
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    lstm = TrafficLSTM(input_size=10, hidden_size=16, output_size=6)

    stopper(1.0, lstm)
    assert stopper.best_loss == 1.0
    assert stopper.counter == 0
    assert not stopper.early_stop

    stopper(1.05, lstm)  # No improvement
    assert stopper.counter == 1
    assert not stopper.early_stop

    stopper(1.04, lstm)  # No improvement, triggers patience=2
    assert stopper.counter == 2
    assert stopper.early_stop


def test_saved_weights_and_scaler_artifacts():
    """Verify that serialized PyTorch weights and scaler load correctly."""
    weights_files = list(Path("models").glob("best_traffic_*.pt"))
    if not weights_files:
        pytest.skip("No PyTorch weights checkpoint found in models/.")

    checkpoint_path = weights_files[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "model_state_dict" in checkpoint
    assert "model_class" in checkpoint
    assert "sequence_features" in checkpoint

    # Reconstruct model from checkpoint
    in_dim = checkpoint["input_size"]
    h_dim = checkpoint["hidden_size"]
    out_dim = checkpoint["output_size"]

    if checkpoint["model_class"] == "LSTM":
        model = TrafficLSTM(input_size=in_dim, hidden_size=h_dim, output_size=out_dim)
    else:
        model = TrafficGRU(input_size=in_dim, hidden_size=h_dim, output_size=out_dim)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_input = torch.randn(2, 24, in_dim)
    with torch.no_grad():
        out = model(sample_input)
    assert out.shape == (2, out_dim)

    # Check scaler
    scaler_path = Path("models/scaler.joblib")
    assert scaler_path.exists()
    scaler = joblib.load(scaler_path)
    assert hasattr(scaler, "mean_")
    assert len(scaler.mean_) == in_dim
