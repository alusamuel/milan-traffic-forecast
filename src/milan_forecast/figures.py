"""Evidence-oriented plots and compact exploratory statistics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import TEST_START, TEST_END


def _finish(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def plot_eda(processed: Path, figures: Path, results: Path | None = None) -> dict:
    figures.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((processed / "manifest.json").read_text())
    totals = pd.read_csv(processed / "area_totals.csv").set_index("square_id").total_internet
    frame = pd.read_pickle(processed / "selected_areas.pkl")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.log1p(totals.values), bins=70, color="#217e9a")
    ax.set(xlabel="log(1 + total Internet activity)", ylabel="Area count",
           title="Distribution over Milan squares (full observation period)")
    _finish(figures / "area_distribution.png")

    first_two_weeks = frame.loc["2013-11-01":"2013-11-14"]
    for area in manifest["analysis_areas"]:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(first_two_weeks.index, first_two_weeks[area], linewidth=0.7)
        ax.set(title=f"Square {area}: first 14 days", ylabel="Internet activity", xlabel="Local time (Europe/Rome)")
        _finish(figures / f"first_two_weeks_area_{area}.png")

    top = manifest["top3"][0]
    s = frame[top].loc[:"2013-12-15 23:59:59"].ffill().fillna(0)
    # Analysis one: lag correlation. Missing slots were filled from prior data.
    # Copy: to_numpy can return a view, and centering in place would corrupt s.
    z = s.to_numpy(dtype=float, copy=True)
    z -= z.mean()
    denom = z @ z
    lags = np.arange(1, 1009)
    acf = np.array([(z[:-lag] @ z[lag:]) / denom if denom else np.nan for lag in lags])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags / 144, acf, color="#217e9a")
    ax.axvline(1, linestyle="--", color="#c98526", label="24 hours")
    ax.axvline(7, linestyle="--", color="#bd4b54", label="7 days")
    ax.set(xlabel="Lag (days)", ylabel="Correlation", title=f"Square {top}: autocorrelation before test week")
    ax.legend()
    _finish(figures / "top_area_acf.png")

    # Analysis two: hourly local-time profile, weekday vs weekend.
    grouped = pd.DataFrame({"activity": s.to_numpy(), "hour": s.index.hour,
                            "weekend": s.index.dayofweek >= 5})
    profile = grouped.groupby(["weekend", "hour"]).activity.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(profile.index, profile[False], label="Weekday", marker=".")
    ax.plot(profile.index, profile[True], label="Weekend", marker=".")
    ax.set(xlabel="Local hour", ylabel="Mean Internet activity",
           title=f"Square {top}: daily profile before test week")
    ax.legend()
    _finish(figures / "top_area_daily_profile.png")
    stats = {"top_area": top, "top3": manifest["top3"],
             "area_total_quantiles": {str(q): float(totals.quantile(q)) for q in [0, .25, .5, .75, .9, .99, 1]},
             "lag_correlation": {str(lag): float(acf[lag - 1]) for lag in [1, 6, 144, 1008]},
             "note": "ACF and daily profile use only pre-test data; past-only filled absent intervals."}
    summary_directory = results if results is not None else figures.parent
    summary_directory.mkdir(parents=True, exist_ok=True)
    (summary_directory / "eda_summary.json").write_text(json.dumps(stats, indent=2))
    return stats


def plot_forecasts(processed: Path, output: Path, figures: Path):
    manifest = json.loads((processed / "manifest.json").read_text())
    figures.mkdir(parents=True, exist_ok=True)
    for area in manifest["evaluation_areas"]:
        prediction = pd.read_csv(output / f"predictions_area_{area}.csv", parse_dates=["local_time"])
        if not len(prediction):
            raise ValueError(f"No test predictions for area {area}")
        for kind in ["ridge", "lstm", "tcn"]:
            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.plot(prediction.local_time, prediction.actual, linewidth=.8, label="Observed", color="#182c46")
            ax.plot(prediction.local_time, prediction[kind], linewidth=.8, label=kind.upper(), color="#dc7532")
            ax.set(title=f"Square {area}: {kind.upper()}, 16-22 December 2013",
                   xlabel="Local time (Europe/Rome)", ylabel="Internet activity")
            ax.legend()
            _finish(figures / f"forecast_{kind}_area_{area}.png")
    _plot_forecast_panel(manifest["evaluation_areas"], output, figures)
    _plot_validation_curves(manifest["evaluation_areas"], output, figures)


def _plot_forecast_panel(areas: list[int], output: Path, figures: Path):
    """All nine forecasts in one 3x3 figure (rows = squares, columns = models) for the report."""
    scores = pd.read_csv(output / "results.csv")
    fig, axes = plt.subplots(len(areas), 3, figsize=(18, 3 * len(areas)), sharex=True, squeeze=False)
    for i, area in enumerate(areas):
        prediction = pd.read_csv(output / f"predictions_area_{area}.csv", parse_dates=["local_time"])
        for j, kind in enumerate(["ridge", "lstm", "tcn"]):
            ax = axes[i, j]
            ax.plot(prediction.local_time, prediction.actual, linewidth=.6, color="#182c46", label="Observed")
            ax.plot(prediction.local_time, prediction[kind], linewidth=.6, color="#dc7532", label=kind.upper())
            mae = scores.loc[(scores.area == area) & (scores.model == kind), "mae"].iloc[0]
            ax.set_title(f"Square {area} - {kind.upper()} (MAE {mae:.1f})", fontsize=10)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            if j == 0:
                ax.set_ylabel("Internet activity")
    axes[0, 0].legend(fontsize=8)
    _finish(figures / "forecast_panel_3x3.png")


def _plot_validation_curves(areas: list[int], output: Path, figures: Path):
    """Per-epoch validation MSE of every neural candidate; dots mark the selected epoch."""
    tuning = json.loads((output / "tuning.json").read_text())
    fig, axes = plt.subplots(1, len(areas), figsize=(5 * len(areas), 3.8), squeeze=False)
    for ax, area in zip(axes[0], areas):
        for item in (t for t in tuning if t["area"] == area and t["epoch_log"]):
            log = pd.DataFrame(item["epoch_log"])
            line, = ax.plot(log.epoch, log.validation_scaled_mse,
                            label=f"{item['model'].upper()} w{item['config']['width']}")
            best = log.loc[log.epoch == item["selected_epoch"]]
            ax.scatter(best.epoch, best.validation_scaled_mse, color=line.get_color(), zorder=3)
        ax.set(title=f"Square {area}", xlabel="Epoch", ylabel="Validation MSE (scaled)", yscale="log")
        ax.legend(fontsize=8)
    _finish(figures / "validation_curves.png")
