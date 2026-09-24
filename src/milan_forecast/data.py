"""Two streaming passes over the original daily Telecom Italia TSV files."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

NAMES = ["square_id", "timestamp_ms", "country_code", "sms_in", "sms_out", "call_in", "call_out", "internet"]
DTYPES = {"square_id": "int32", "timestamp_ms": "int64", "internet": "float64"}
START = pd.Timestamp("2013-11-01", tz="Europe/Rome")
END = pd.Timestamp("2014-01-01", tz="Europe/Rome")
TIMES = pd.date_range(START, END, freq="10min", inclusive="left")


def raw_files(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.rglob("*") if p.is_file() and
                   (p.name.endswith(".txt") or p.name.endswith(".txt.gz") or p.name.endswith(".tsv")))
    if not files:
        raise FileNotFoundError(f"No daily .txt/.tsv/.txt.gz files in {directory}. See README.md.")
    expected = {d.strftime("%Y-%m-%d") for d in pd.date_range("2013-11-01", "2013-12-31")}
    dates = [match.group(1) for p in files if (match := re.search(r"(2013-(?:11|12)-\d{2})", p.name))]
    if len(dates) != len(files) or set(dates) != expected or len(dates) != len(set(dates)):
        missing = sorted(expected - set(dates))
        duplicates = sorted({d for d in dates if dates.count(d) > 1})
        raise ValueError(f"Expected one Milan daily activity file for every date from Nov 1 through Dec 31 2013 (61 files). Found {len(files)}; missing={missing}; duplicate dates={duplicates}. Keep unrelated files elsewhere.")
    return files


def chunks(files: list[Path], chunk_size: int):
    for file in files:
        for chunk in pd.read_csv(file, sep="\t", header=None, names=NAMES,
                                 usecols=[0, 1, 7], dtype=DTYPES,
                                 chunksize=chunk_size, compression="infer"):
            yield file.name, chunk


def current_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        # A Linux development shell can test the data code before installing extras.
        status = Path("/proc/self/status")
        if status.exists():
            line = next(line for line in status.read_text().splitlines() if line.startswith("VmRSS:"))
            return int(line.split()[1]) / 1024
        raise RuntimeError("Install psutil to measure current process memory on this operating system")


def rank_areas(files: list[Path], chunk_size: int) -> tuple[pd.Series, dict]:
    totals = np.zeros(10001, dtype=np.float64)
    rows = 0
    for _, frame in chunks(files, chunk_size):
        if frame.square_id.isna().any() or frame.timestamp_ms.isna().any():
            raise ValueError("Missing square/time key in raw data")
        ids = frame.square_id.to_numpy()
        if ((ids < 1) | (ids > 10000)).any():
            raise ValueError("Square ID outside 1..10000")
        # Missing Internet fields in some country rows mean zero Internet events.
        internet = frame.internet.fillna(0).to_numpy(dtype=np.float64)
        if (internet < 0).any():
            raise ValueError("Negative Internet activity")
        np.add.at(totals, ids, internet)
        rows += len(frame)
    series = pd.Series(totals[1:], index=np.arange(1, 10001), name="total_internet")
    # Stable tie break: smaller ID wins.
    series = series.sort_values(ascending=False, kind="stable")
    evidence = {"rows_scanned": rows, "chunk_size": chunk_size,
                "rss_mb_after_scan": current_rss_mb(),
                "totals_array_bytes": totals.nbytes,
                "memory_measurement_note": "RSS after scan is process-wide current memory, not isolated chunk memory or process peak; baseline is measured before scan. Full frame is measured separately by memory_probe."}
    return series, evidence


def memory_probe(file: Path, rows: int = 100_000) -> dict:
    """Measure same-row in-memory pandas frames; do not compare synthetic estimates."""
    full = pd.read_csv(file, sep="\t", header=None, names=NAMES, nrows=rows,
                       compression="infer", low_memory=False)
    full_bytes = int(full.memory_usage(deep=True).sum())
    count = len(full)
    del full
    slim = pd.read_csv(file, sep="\t", header=None, names=NAMES, usecols=[0,1,7],
                       dtype=DTYPES, nrows=rows, compression="infer")
    slim_bytes = int(slim.memory_usage(deep=True).sum())
    return {"file": file.name, "rows": count, "full_8_column_bytes": full_bytes,
            "selected_3_column_bytes": slim_bytes,
            "reduction_percent": 100 * (1 - slim_bytes / full_bytes)}


def extract_areas(files: list[Path], area_ids: list[int], chunk_size: int) -> tuple[pd.DataFrame, dict]:
    """Aggregate all country codes per square and 10-minute interval; observed slots only."""
    values = np.zeros((len(TIMES), len(area_ids)), dtype=np.float64)
    seen = np.zeros_like(values, dtype=np.int32)
    lookup = {area: j for j, area in enumerate(area_ids)}
    for _, frame in chunks(files, chunk_size):
        selected = frame.loc[frame.square_id.isin(area_ids)]
        if selected.empty:
            continue
        dt = pd.to_datetime(selected.timestamp_ms, unit="ms", utc=True).dt.tz_convert("Europe/Rome")
        idx = TIMES.get_indexer(dt)
        cols = selected.square_id.map(lookup).to_numpy(dtype=np.int32)
        valid = idx >= 0
        if valid.any():
            np.add.at(values, (idx[valid], cols[valid]), selected.internet.fillna(0).to_numpy()[valid])
            np.add.at(seen, (idx[valid], cols[valid]), 1)
    # A completely absent square/time is unknown, not automatically zero.
    values[seen == 0] = np.nan
    output = pd.DataFrame(values, index=TIMES, columns=area_ids)
    evidence = {str(a): {"observed_intervals": int((seen[:, j] > 0).sum()),
                         "missing_intervals": int((seen[:, j] == 0).sum())}
                for j, a in enumerate(area_ids)}
    return output, evidence


def prepare(raw_dir: Path, out_dir: Path, chunk_size: int = 500_000) -> dict:
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    files = raw_files(raw_dir)
    initial_rss = current_rss_mb()
    ranking, evidence = rank_areas(files, chunk_size)
    top3 = [int(i) for i in ranking.index[:3]]
    chosen = list(dict.fromkeys(top3 + [4159, 4556]))
    frame, coverage = extract_areas(files, chosen, chunk_size)
    # Keep raw data outside Git; rerun preparation after changing source files.
    ranking.to_csv(out_dir / "area_totals.csv", header=True, index_label="square_id")
    frame.to_pickle(out_dir / "selected_areas.pkl")
    manifest = {"files": [p.name for p in files], "top3": top3,
                "evaluation_areas": list(dict.fromkeys([top3[0], 4159, 4556])),
                "analysis_areas": chosen, "coverage": coverage, "scan": evidence,
                "rss_before_mb": initial_rss, "rss_after_mb": current_rss_mb(),
                "memory_probe": memory_probe(files[0]),
                "calendar_timezone": "Europe/Rome", "missing_interval_policy": "unknown; limited past-only fill during experiments"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
