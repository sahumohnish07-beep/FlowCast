"""Deep learning sequence models (LSTM and GRU) for multi-horizon traffic flow forecasting.

Implements PyTorch recurrent neural networks using a 24-hour sliding sequence window
to predict traffic volume across the next 1-6 hours (multi-horizon output).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys

# Ensure FlowCast root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("flowcast.models.lstm")

# Features utilized in sequence representation (traffic volume must be at index 0)
SEQUENCE_FEATURES = [
    "traffic_volume",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "is_rush_hour",
    "temp_c",
    "weather_severity_index",
    "relative_humidity_2m",
]


class EarlyStopping:
    """Early stopping handler to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state: dict[str, torch.Tensor] | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Clone model state dict
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model: nn.Module) -> None:
        """Load best state dict into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class TrafficLSTM(nn.Module):
    """Deep Long Short-Term Memory network for multi-horizon traffic forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 48,
        num_layers: int = 1,
        output_size: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # (batch_size, hidden_size)
        h = self.relu(self.fc1(last_step))
        return self.fc2(h)  # (batch_size, output_size)


class TrafficGRU(nn.Module):
    """Gated Recurrent Unit network for multi-horizon traffic forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 48,
        num_layers: int = 1,
        output_size: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        h = self.relu(self.fc1(last_step))
        return self.fc2(h)


def create_sliding_windows(
    scaled_features: np.ndarray,
    seq_len: int = 24,
    horizon: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct sliding windows of (seq_len) input hours to (horizon) target hours.

    Args:
        scaled_features: 2D numpy array where column 0 is the scaled traffic volume.
        seq_len: Number of historical lag timesteps (24 hours).
        horizon: Number of forward prediction timesteps (6 hours).

    Returns:
        Tuple of (X, y) where X shape is (N, seq_len, num_features) and y shape is (N, horizon).
    """
    num_samples = len(scaled_features) - seq_len - horizon + 1
    if num_samples <= 0:
        raise ValueError("Insufficient rows for sequence length and horizon.")

    X_list = []
    y_list = []

    target_series = scaled_features[:, 0]  # First column is traffic_volume

    for i in range(num_samples):
        X_list.append(scaled_features[i : i + seq_len])
        y_list.append(target_series[i + seq_len : i + seq_len + horizon])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def prepare_dataloaders(
    processed_dir: str | Path = "data/processed",
    seq_len: int = 24,
    horizon: int = 6,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler, pd.Series]:
    """Load splits, scale features, construct sliding windows, and return PyTorch DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, scaler, test_timestamps)
    """
    proc = Path(processed_dir).resolve()
    df_train = pd.read_parquet(proc / "train.parquet")
    df_val = pd.read_parquet(proc / "val.parquet")
    df_test = pd.read_parquet(proc / "test.parquet")

    # Fit scaler strictly on training split
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train[SEQUENCE_FEATURES])
    val_scaled = scaler.transform(df_val[SEQUENCE_FEATURES])
    test_scaled = scaler.transform(df_test[SEQUENCE_FEATURES])

    # Build sequence windows
    X_train, y_train = create_sliding_windows(train_scaled, seq_len, horizon)
    X_val, y_val = create_sliding_windows(val_scaled, seq_len, horizon)
    X_test, y_test = create_sliding_windows(test_scaled, seq_len, horizon)

    test_timestamps = df_test["date_time"].iloc[seq_len : seq_len + len(X_test)].reset_index(drop=True)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        "DataLoaders prepared: Train windows=%d, Val windows=%d, Test windows=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    return train_loader, val_loader, test_loader, scaler, test_timestamps


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    epochs: int = 8,
    lr: float = 0.003,
    patience: int = 3,
    device: str = "cpu",
) -> tuple[nn.Module, list[float], list[float]]:
    """Train a neural network with early stopping and learning rate scheduling."""
    logger.info("Training %s for max %d epochs...", model_name, epochs)
    dev = torch.device(device)
    model.to(dev)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    early_stopping = EarlyStopping(patience=patience)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(batch_x)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                running_val_loss += loss.item() * len(batch_x)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        logger.info(
            "[%s] Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f",
            model_name,
            epoch,
            epochs,
            epoch_train_loss,
            epoch_val_loss,
        )

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            logger.info("[%s] Early stopping triggered at epoch %d.", model_name, epoch)
            break

    early_stopping.restore_best_weights(model)
    return model, train_losses, val_losses


def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    """Predict on a DataLoader and compute unscaled 1-step and multi-horizon metrics.

    Returns:
        (y_true_unscaled, y_pred_unscaled, step1_metrics, avg_horizon_metrics)
    """
    dev = torch.device(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(dev)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    y_pred_scaled = np.vstack(all_preds)
    y_true_scaled = np.vstack(all_targets)

    # Invert scaling: traffic_volume is at column index 0
    target_mean = float(scaler.mean_[0])
    target_scale = float(scaler.scale_[0])

    y_pred = np.clip(y_pred_scaled * target_scale + target_mean, 0.0, None)
    y_true = np.clip(y_true_scaled * target_scale + target_mean, 0.0, None)

    # 1. Horizon 1 (Next 1-hour ahead, directly aligned with Phase 3 baseline evaluation)
    h1_true = y_true[:, 0]
    h1_pred = y_pred[:, 0]
    step1_metrics = {
        "MAE": round(float(mean_absolute_error(h1_true, h1_pred)), 2),
        "RMSE": round(float(root_mean_squared_error(h1_true, h1_pred)), 2),
        "R2": round(float(r2_score(h1_true, h1_pred)), 4),
    }

    # 2. Multi-horizon average across 1-6 hours
    mae_all = float(mean_absolute_error(y_true.flatten(), y_pred.flatten()))
    rmse_all = float(root_mean_squared_error(y_true.flatten(), y_pred.flatten()))
    r2_all = float(r2_score(y_true.flatten(), y_pred.flatten()))

    avg_horizon_metrics = {
        "MAE": round(mae_all, 2),
        "RMSE": round(rmse_all, 2),
        "R2": round(r2_all, 4),
    }

    return y_true, y_pred, step1_metrics, avg_horizon_metrics


def plot_loss_curves(
    lstm_train: list[float],
    lstm_val: list[float],
    gru_train: list[float],
    gru_val: list[float],
    output_path: str | Path = "docs/images/lstm_training_loss.png",
) -> Path:
    """Export training and validation loss convergence curves for LSTM and GRU."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # LSTM Subplot
    axes[0].plot(range(1, len(lstm_train) + 1), lstm_train, label="Train Loss", marker="o", color="#1f77b4")
    axes[0].plot(range(1, len(lstm_val) + 1), lstm_val, label="Val Loss", marker="s", linestyle="--", color="#ff7f0e")
    axes[0].set_title("Traffic LSTM: Huber Loss Convergence", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # GRU Subplot
    axes[1].plot(range(1, len(gru_train) + 1), gru_train, label="Train Loss", marker="o", color="#2ca02c")
    axes[1].plot(range(1, len(gru_val) + 1), gru_val, label="Val Loss", marker="s", linestyle="--", color="#d62728")
    axes[1].set_title("Traffic GRU: Huber Loss Convergence", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    logger.info("Saved loss convergence curves to %s", out)
    return out


def plot_forecast_comparison(
    timestamps: pd.Series,
    y_true: np.ndarray,
    lstm_pred: np.ndarray,
    gru_pred: np.ndarray,
    output_path: str | Path = "docs/images/lstm_gru_forecast.png",
    zoom_hours: int = 168,
) -> Path:
    """Plot multi-horizon forecast comparisons against ground truth."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5.5), dpi=300)
    time_slice = timestamps.iloc[:zoom_hours]

    plt.plot(time_slice, y_true[:zoom_hours, 0], label="Actual Volume (t+1)", color="black", linewidth=2.0, alpha=0.85)
    plt.plot(time_slice, lstm_pred[:zoom_hours, 0], label="LSTM (t+1)", color="#1f77b4", linewidth=1.8, linestyle="--")
    plt.plot(time_slice, gru_pred[:zoom_hours, 0], label="GRU (t+1)", color="#2ca02c", linewidth=1.8, linestyle=":")

    plt.title(f"Deep Sequence Multi-Horizon Forecast: Actual vs. LSTM vs. GRU (First {zoom_hours}h of 2018 Test Data)", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Observation Timestamp")
    plt.ylabel("Traffic Volume (vehicles/hour)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    logger.info("Saved forecast comparison plot to %s", out)
    return out


def update_comparison_table(
    new_results: list[dict[str, object]],
    baseline_path: str | Path = "docs/baseline_results.csv",
) -> pd.DataFrame:
    """Append deep learning metrics to the baseline comparison table and save."""
    base_file = Path(baseline_path).resolve()
    if base_file.exists():
        df_base = pd.read_csv(base_file)
        # Filter out old LSTM/GRU rows if re-running
        df_base = df_base[~df_base["Model"].isin(["LSTM (1h)", "GRU (1h)", "LSTM", "GRU"])]
    else:
        df_base = pd.DataFrame()

    df_new = pd.DataFrame(new_results)
    df_combined = pd.concat([df_base, df_new], ignore_index=True)
    df_combined = df_combined.sort_values(by="Test_RMSE").reset_index(drop=True)
    df_combined.to_csv(base_file, index=False)
    logger.info("Updated model comparison table in %s", base_file)
    return df_combined


def run_deep_learning_pipeline(
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "models",
    docs_dir: str | Path = "docs",
    epochs: int = 8,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, str]:
    """Execute sequence data loading, LSTM & GRU training, evaluation, and artifact saving."""
    logger.info("Initiating Phase 4 Deep Learning Pipeline (LSTM & GRU)...")

    # Step 1: Prepare sliding window sequences
    train_loader, val_loader, test_loader, scaler, test_timestamps = prepare_dataloaders(
        processed_dir=processed_dir,
        seq_len=24,
        horizon=6,
        batch_size=batch_size,
    )

    in_features = len(SEQUENCE_FEATURES)

    # Step 2: Train LSTM
    lstm_model = TrafficLSTM(input_size=in_features, hidden_size=48, num_layers=1, output_size=6)
    lstm_model, lstm_tr_loss, lstm_val_loss = train_model(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="LSTM",
        epochs=epochs,
        lr=0.003,
        patience=3,
    )

    # Step 3: Train GRU
    gru_model = TrafficGRU(input_size=in_features, hidden_size=48, num_layers=1, output_size=6)
    gru_model, gru_tr_loss, gru_val_loss = train_model(
        model=gru_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="GRU",
        epochs=epochs,
        lr=0.003,
        patience=3,
    )

    # Step 4: Evaluate Models on Test Set
    y_true_lstm, lstm_pred, lstm_s1, lstm_multi = evaluate_predictions(lstm_model, test_loader, scaler)
    y_true_gru, gru_pred, gru_s1, gru_multi = evaluate_predictions(gru_model, test_loader, scaler)

    # Validation evaluation for reporting
    _, _, lstm_val_s1, _ = evaluate_predictions(lstm_model, val_loader, scaler)
    _, _, gru_val_s1, _ = evaluate_predictions(gru_model, val_loader, scaler)

    # Step 5: Save visualizations
    docs_path = Path(docs_dir).resolve()
    img_dir = docs_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    loss_plot_path = img_dir / "lstm_training_loss.png"
    plot_loss_curves(lstm_tr_loss, lstm_val_loss, gru_tr_loss, gru_val_loss, output_path=loss_plot_path)

    forecast_plot_path = img_dir / "lstm_gru_forecast.png"
    plot_forecast_comparison(test_timestamps, y_true_lstm, lstm_pred, gru_pred, output_path=forecast_plot_path)

    # Step 6: Update comparison table
    new_metrics = [
        {
            "Model": "LSTM (1h)",
            "Test_MAE": lstm_s1["MAE"],
            "Test_RMSE": lstm_s1["RMSE"],
            "Test_R2": lstm_s1["R2"],
            "Val_MAE": lstm_val_s1["MAE"],
            "Val_RMSE": lstm_val_s1["RMSE"],
            "Val_R2": lstm_val_s1["R2"],
            "Train_MAE": None,
            "Train_RMSE": None,
            "Train_R2": None,
        },
        {
            "Model": "GRU (1h)",
            "Test_MAE": gru_s1["MAE"],
            "Test_RMSE": gru_s1["RMSE"],
            "Test_R2": gru_s1["R2"],
            "Val_MAE": gru_val_s1["MAE"],
            "Val_RMSE": gru_val_s1["RMSE"],
            "Val_R2": gru_val_s1["R2"],
            "Train_MAE": None,
            "Train_RMSE": None,
            "Train_R2": None,
        },
    ]

    df_comparison = update_comparison_table(new_metrics, baseline_path=docs_path / "baseline_results.csv")

    # Step 7: Save best deep model weights and scaler
    best_deep_name = "LSTM" if lstm_s1["RMSE"] <= gru_s1["RMSE"] else "GRU"
    best_deep_model = lstm_model if best_deep_name == "LSTM" else gru_model

    models_path = Path(models_dir).resolve()
    models_path.mkdir(parents=True, exist_ok=True)

    model_weights_path = models_path / f"best_traffic_{best_deep_name.lower()}.pt"
    torch.save(
        {
            "model_state_dict": best_deep_model.state_dict(),
            "model_class": best_deep_name,
            "input_size": in_features,
            "hidden_size": 48,
            "num_layers": 1,
            "output_size": 6,
            "sequence_features": SEQUENCE_FEATURES,
            "step1_test_metrics": lstm_s1 if best_deep_name == "LSTM" else gru_s1,
            "saved_at": datetime.now().isoformat(),
        },
        model_weights_path,
    )

    scaler_path = models_path / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    logger.info("Saved %s weights (%.1f KB) to %s", best_deep_name, model_weights_path.stat().st_size / 1024, model_weights_path)
    logger.info("Saved feature & target scaler (%.1f KB) to %s", scaler_path.stat().st_size / 1024, scaler_path)

    print("\n" + "=" * 80)
    print(" FlowCast Phase 4: Deep Learning Performance Comparison (Test Set)")
    print("=" * 80)
    print(f"LSTM (1h step): MAE = {lstm_s1['MAE']}, RMSE = {lstm_s1['RMSE']}, R² = {lstm_s1['R2']}")
    print(f"LSTM (6h avg) : MAE = {lstm_multi['MAE']}, RMSE = {lstm_multi['RMSE']}, R² = {lstm_multi['R2']}")
    print(f"GRU  (1h step): MAE = {gru_s1['MAE']}, RMSE = {gru_s1['RMSE']}, R² = {gru_s1['R2']}")
    print(f"GRU  (6h avg) : MAE = {gru_multi['MAE']}, RMSE = {gru_multi['RMSE']}, R² = {gru_multi['R2']}")
    print("=" * 80)
    print(f"Full Leaderboard:\n{df_comparison[['Model', 'Test_MAE', 'Test_RMSE', 'Test_R2']].to_string(index=False)}")
    print("=" * 80 + "\n")

    return df_comparison, best_deep_name


def main() -> None:
    """CLI entrypoint for deep learning models."""
    parser = argparse.ArgumentParser(description="Train deep learning LSTM/GRU models for FlowCast.")
    parser.add_argument("--epochs", type=int, default=8, help="Maximum epochs to train.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_deep_learning_pipeline(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

