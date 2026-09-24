"""Chronological tuning, final fitting, and observed-history one-step prediction."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import START, TIMES

TRAIN_END = pd.Timestamp("2013-12-09", tz="Europe/Rome")
TEST_START = pd.Timestamp("2013-12-16", tz="Europe/Rome")
TEST_END = pd.Timestamp("2013-12-23", tz="Europe/Rome")
WINDOW = 144  # preceding 24 hours at 10-minute resolution
SEED = 117
# Iteration 1 used 8; its validation logs peaked at the cap, so iteration 2 allows 40.
MAX_EPOCHS = 40


def series_and_validity(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    valid = s.notna().to_numpy()
    # Only earlier data are used for a missing history observation.
    filled = s.ffill().fillna(0).to_numpy(dtype=np.float64)
    if (filled < 0).any():
        raise ValueError("Negative Internet activity")
    return filled, valid


def split_windows(s: pd.Series, window: int = WINDOW):
    values, valid = series_and_validity(s)
    X = np.lib.stride_tricks.sliding_window_view(values, window)[:-1].copy()
    y = values[window:]
    target_times = s.index[window:]
    # Window at t uses indices [t-window, t-1], never the target at t.
    keep = valid[window:]
    splits = {"train": keep & (target_times < TRAIN_END),
              "val": keep & (target_times >= TRAIN_END) & (target_times < TEST_START),
              "pretest": keep & (target_times < TEST_START),
              "test": keep & (target_times >= TEST_START) & (target_times < TEST_END)}
    if not splits["train"].any() or not splits["val"].any() or not splits["test"].any():
        raise ValueError("Insufficient observed target intervals in chronological train/validation/test split")
    return X, y, target_times, splits


class LogStandardizer:
    def fit(self, values: np.ndarray):
        z = np.log1p(values.astype(np.float64))
        self.mean = float(np.mean(z))
        self.std = max(float(np.std(z)), 1e-8)
        return self

    def transform(self, values):
        return ((np.log1p(values) - self.mean) / self.std).astype(np.float32)

    def inverse(self, values):
        return np.maximum(np.expm1(np.asarray(values, dtype=np.float64) * self.std + self.mean), 0)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    if len(y) != len(pred) or not len(y):
        raise ValueError("Empty or unequal metric arrays")
    positive = y > 0
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mape_percent": float(np.mean(np.abs((y[positive] - pred[positive]) / y[positive])) * 100) if positive.any() else None,
            "mape_nonzero_count": int(positive.sum()), "n": int(len(y))}


def _torch_components():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for LSTM and TCN; install with 'pip install -e .' ") from exc

    class LSTM(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.encoder = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.readout = nn.Linear(hidden, 1)

        def forward(self, x):
            _, (h, _) = self.encoder(x.unsqueeze(-1))
            return self.readout(h[-1]).squeeze(-1)

    class CausalConv(nn.Module):
        def __init__(self, channels, dilation):
            super().__init__()
            self.pad = 2 * dilation
            self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation, padding=self.pad)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x)[..., :-self.pad])

    class TCN(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.entry = nn.Conv1d(1, channels, 1)
            self.blocks = nn.Sequential(*(CausalConv(channels, 2 ** i) for i in range(6)))
            self.readout = nn.Linear(channels, 1)

        def forward(self, x):
            z = self.blocks(self.entry(x.unsqueeze(1)))
            return self.readout(z[..., -1]).squeeze(-1)

    return torch, LSTM, TCN


def _fit_neural(kind: str, cfg: dict, x: np.ndarray, y: np.ndarray,
                epochs: int, validation: tuple[np.ndarray, np.ndarray] | None = None):
    torch, LSTM, TCN = _torch_components()
    torch.manual_seed(SEED)
    # CPU by default makes measurements comparable across machines.
    device = torch.device("cpu")
    model = LSTM(cfg["width"]) if kind == "lstm" else TCN(cfg["width"])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = torch.nn.MSELoss()
    x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
    best_loss, best_epoch, best_state = float("inf"), epochs, None
    epoch_log = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        gen = torch.Generator().manual_seed(SEED + epoch)
        for indices in torch.randperm(len(x_t), generator=gen).split(256):
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x_t[indices]), y_t[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if validation is not None:
            pred = _predict_neural(model, validation[0])
            val_loss = float(np.mean((pred - validation[1]) ** 2))
            epoch_log.append({"epoch": epoch, "validation_scaled_mse": val_loss})
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fit_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, fit_seconds, epoch_log


def _predict_neural(model, x: np.ndarray) -> np.ndarray:
    import torch
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(batch)).numpy() for batch in np.array_split(x, max(1, (len(x) + 511) // 512))])


def run_area(s: pd.Series, area: int, models_dir: Path | None = None,
             max_epochs: int = MAX_EPOCHS) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    X, y, target_times, splits = split_windows(s)
    tuner = LogStandardizer().fit(y[splits["train"]])
    x_tune, y_tune = tuner.transform(X), tuner.transform(y)
    final_scaler = LogStandardizer().fit(y[splits["pretest"]])
    x_final, y_final = final_scaler.transform(X), final_scaler.transform(y)
    # Prespecified small grid; validation is never consulted on the test week.
    configs = {"ridge": [{"alpha": 0.1}, {"alpha": 10.0}],
               "lstm": [{"width": 16, "lr": 0.001}, {"width": 32, "lr": 0.001}],
               "tcn": [{"width": 16, "lr": 0.001}, {"width": 32, "lr": 0.001}]}
    records, predictions, tuning = [], pd.DataFrame(index=target_times[splits["test"]]), []
    predictions.index.name = "local_time"
    predictions["actual"] = y[splits["test"]]
    for kind, candidates in configs.items():
        scores = []
        for cfg in candidates:
            if kind == "ridge":
                candidate = Ridge(alpha=cfg["alpha"]).fit(x_tune[splits["train"]], y_tune[splits["train"]])
                pred_scaled = candidate.predict(x_tune[splits["val"]])
                best_epoch, epoch_log = None, []
            else:
                candidate, best_epoch, _, epoch_log = _fit_neural(
                    kind, cfg, x_tune[splits["train"]], y_tune[splits["train"]], epochs=max_epochs,
                    validation=(x_tune[splits["val"]], y_tune[splits["val"]]))
                pred_scaled = _predict_neural(candidate, x_tune[splits["val"]])
            val = metrics(y[splits["val"]], tuner.inverse(pred_scaled))
            item = {"area": area, "model": kind, "config": cfg, "validation_metrics": val,
                    "selected_epoch": best_epoch, "max_epochs": max_epochs, "epoch_log": epoch_log,
                    "selection_rule": "minimum validation MAE, first candidate on tie"}
            tuning.append(item)
            scores.append((val["mae"], cfg, best_epoch))
        _, chosen_cfg, chosen_epoch = min(scores, key=lambda row: row[0])
        t0 = time.perf_counter()
        if kind == "ridge":
            model = Ridge(alpha=chosen_cfg["alpha"]).fit(x_final[splits["pretest"]], y_final[splits["pretest"]])
        else:
            model, _, _, _ = _fit_neural(kind, chosen_cfg, x_final[splits["pretest"]],
                                         y_final[splits["pretest"]], epochs=chosen_epoch)
        training_seconds = time.perf_counter() - t0
        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            identifier = f"{kind}_area_{area}"
            if kind == "ridge":
                np.savez_compressed(models_dir / f"{identifier}.npz",
                                    coef=model.coef_, intercept=model.intercept_)
            else:
                import torch
                torch.save(model.state_dict(), models_dir / f"{identifier}.pt")
            (models_dir / f"{identifier}.json").write_text(json.dumps({
                "model": kind, "area": area, "configuration": chosen_cfg,
                "final_epochs": chosen_epoch, "max_epochs": max_epochs, "window_intervals": WINDOW,
                "log_scaler_mean": final_scaler.mean, "log_scaler_std": final_scaler.std,
                "note": "Only load model artifacts from sources you trust. See README for provenance."
            }, indent=2))
        t0 = time.perf_counter()
        scaled = model.predict(x_final[splits["test"]]) if kind == "ridge" else _predict_neural(model, x_final[splits["test"]])
        prediction_seconds = time.perf_counter() - t0
        pred = final_scaler.inverse(scaled)
        predictions[kind] = pred
        records.append({"area": area, "model": kind, **metrics(y[splits["test"]], pred),
                        "chosen_config": json.dumps(chosen_cfg), "epochs_final": chosen_epoch, "max_epochs": max_epochs,
                        "final_fit_seconds": training_seconds, "week_inference_seconds": prediction_seconds,
                        "window_intervals": WINDOW, "train_targets": int(splits["train"].sum()),
                        "val_targets": int(splits["val"].sum()),
                        "final_fit_targets": int(splits["pretest"].sum())})
    return records, predictions, tuning


def run_all(processed: Path, out: Path, models_dir: Path | None = None,
            max_epochs: int = MAX_EPOCHS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((processed / "manifest.json").read_text())
    frame = pd.read_pickle(processed / "selected_areas.pkl")
    all_results, all_tuning = [], []
    if len(manifest["evaluation_areas"]) != 3:
        raise ValueError("Evaluation requires three distinct areas; adjust area selection in manifest")
    for area in manifest["evaluation_areas"]:
        result, predictions, tuning = run_area(frame[area], area, models_dir, max_epochs)
        all_results.extend(result)
        all_tuning.extend(tuning)
        predictions.to_csv(out / f"predictions_area_{area}.csv")
        pd.DataFrame(result)[["model", "mae", "mape_percent", "rmse", "mape_nonzero_count", "n"]].to_csv(
            out / f"metrics_area_{area}.csv", index=False)
        errors = []
        for model_name in ["ridge", "lstm", "tcn"]:
            worst = (predictions[model_name] - predictions.actual).abs().idxmax()
            errors.append({"area": area, "model": model_name, "local_time": str(worst),
                           "actual": float(predictions.loc[worst, "actual"]),
                           "predicted": float(predictions.loc[worst, model_name]),
                           "absolute_error": float(abs(predictions.loc[worst, model_name] - predictions.loc[worst, "actual"]))})
        pd.DataFrame(errors).to_csv(out / f"worst_errors_area_{area}.csv", index=False)
    pd.DataFrame(all_results).to_csv(out / "results.csv", index=False)
    (out / "tuning.json").write_text(json.dumps(all_tuning, indent=2))
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "unavailable"
    hardware = {"processor": platform.processor(), "machine": platform.machine(),
                "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(),
                "python": platform.python_version(), "pytorch": torch_version, "max_epochs": max_epochs,
                "timing": "time.perf_counter wall clock; per-area final fit and 1008-slot week inference; excludes tuning, input loading, and plotting; CPU, one run, no uncertainty interval"}
    (out / "hardware.json").write_text(json.dumps(hardware, indent=2))
    return hardware
