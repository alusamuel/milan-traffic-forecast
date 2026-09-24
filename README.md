# Milan mobile Internet traffic forecasting

A research workflow for **Formative Assignment 1: Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting**. It compares a linear autoregressive Ridge model, an LSTM, and a causal temporal convolutional network (TCN) for one-step-ahead forecasts of Internet activity at 10-minute resolution.

**Status:** the full pipeline has been run on the complete 61-day Milan data (314,966,126 raw rows, 1 Nov-31 Dec 2013) on CPU. Generated evidence is in `data/processed/`, `results/results/`, `figures/figures/` and `models/models/`, and headline numbers are summarised under *Results summary* below. The [23-page PDF documentation](report/Milan_Traffic_Forecasting_Documentation.pdf) includes the empirical report, all required plots and tables, reproduction instructions and requirement traceability. The [Markdown companion](report/PROJECT_DOCUMENTATION.md) is also available. Author details, the individual video and actual GitHub/video URLs remain outstanding. The student should inspect, explain and revise the work, and personalize the included AI disclosure.

## Directory map

```text
milan-traffic-forecast/
├── data/
│   ├── raw/                     original 61 daily Milan files (user supplies)
│   └── processed/               selected series and ranking (generated)
├── figures/figures/             EDA and nine forecast plots (generated)
├── models/models/               fitted Ridge/LSTM/TCN artifacts (generated)
├── notebooks/notebooks/         full self-contained pipeline notebook + results viewer
├── results/results/             metrics, predictions, tuning, timing (generated)
├── .gitignore
├── README.md                    setup, run order, assumptions
├── requirements.txt             installable dependency list
├── run.py                       convenient stage runner
├── pyproject.toml               package metadata and CLI
├── src/milan_forecast/
│   ├── data.py                  two-pass chunked aggregation and memory measurements
│   ├── experiments.py           chronological tuning, fitting, metrics, timing
│   ├── figures.py               five EDA series, distribution, two analyses, nine forecasts
│   └── cli.py                   stage runner
├── tests/                       aggregation, timezone, leakage, metric checks
├── report/                      report drafting guidance
└── docs/                        method, sources, video, and requirement map
```

## Source data

Obtain the **Telecommunications - SMS, Call, Internet - MI** daily files from [Harvard Dataverse, DOI 10.7910/DVN/EGZHFV](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV). Extract every Milan activity `.txt` file for **1 November through 31 December 2013** into `data/raw/`. A file can also be `.txt.gz` or `.tsv`. Do not put the other Milan mobility/connectivity datasets or Trentino data in this directory. The original dataset is not redistributed here.

Expected input: eight **headerless, tab-delimited** columns in this order: square ID, Unix timestamp in milliseconds, country code, SMS in, SMS out, calls in, calls out, Internet activity. Multiple country-code records in the same square and interval are summed. Times are converted from UTC timestamps to `Europe/Rome`; the evaluation week is local **16 December 00:00 through 22 December 23:50**, 1008 target slots if fully observed. The Internet field is activity derived from CDRs; it is not necessarily bytes transferred or a literal user count. Check a raw file against this schema before trusting results.

Preparation requires one file per date for all 61 days and rejects incomplete or duplicate collections. Check `data/processed/manifest.json` for missing intervals after processing.

## Setup

Use Python 3.10+ in a fresh environment. PyTorch is needed for the LSTM and TCN; its installation can require a large download. For an environment-specific PyTorch wheel, use the [official installation selector](https://pytorch.org/get-started/locally/) before installing this package.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v   # tests add src/ to the path themselves
```

From the project root:

```bash
python run.py prepare --raw data/raw --chunk-size 500000
python run.py eda
python run.py train                   # --max-epochs 40 by default; 8 reproduces tuning iteration 1
python run.py plots
```

`python run.py all` runs all stages in order. **Alternatively**, open `notebooks/notebooks/mobile_traffic_forecasting.ipynb` with the `.venv` kernel and *Run All*. It contains the same workflow inline, with explanations, and writes the same outputs. Optionally, `python -m pip install -e . --no-deps` installs the equivalent `milan-forecast` command. A smaller `--chunk-size` limits peak memory further at the cost of more read overhead. The work uses CPU by default. Training six neural candidates per area may take a long time; make time measurements on the hardware actually used for the report. On the reference machine (6-core Intel CPU, Windows 11) preparation took about 6 minutes and training with `--max-epochs 40` about 15-20 minutes. Metrics and predictions are bit-identical between runs (fixed seed, CPU), but wall-clock times vary by a few seconds.

The Dataverse files require a guestbook response (an email address) before download. The web interface asks for it; through the API, POST `{"guestbookResponse": {"email": "...", "answers": []}}` to `/api/access/datafile/<id>` to get a signed URL. Skip the `2014-01-01` file.

## Outputs and how to inspect them

| Path | Meaning |
| --- | --- |
| `data/processed/manifest.json` | raw file list, full-period top three, three evaluation areas, missing intervals, memory probe and process RSS snapshots |
| `data/processed/area_totals.csv` | full-period totals of all 10,000 squares |
| `results/results/eda_summary.json` and `figures/figures/*` | distribution, five first-two-week time series, ACF and weekday/weekend daily profile |
| `results/results/tuning.json` | every candidate, validation metrics, per-epoch neural validation MSE and selected epoch |
| `results/results/results.csv` | 3 areas × 3 models: MAE, RMSE, MAPE, nonzero denominator count, times, window, target counts |
| `results/results/metrics_area_<id>.csv` | one required metric table per evaluation area |
| `results/results/worst_errors_area_<id>.csv` | exact largest absolute-error point per model to investigate |
| `results/results/predictions_area_<id>.csv` | timestamps, actual and all three predicted series |
| `figures/figures/forecast_*` | nine actual-versus-predicted plots, plus `forecast_panel_3x3.png` combining them |
| `figures/figures/validation_curves.png` | per-epoch validation MSE of every neural candidate |
| `results/results/hardware.json` | machine and timing measurement method |
| `models/models/<model>_area_<id>.*` | final fitted parameters and scaler/config metadata; load only trusted artifacts |

Read `docs/methodology.md` for the rationale and limitations. The completed documentation is in `report/Milan_Traffic_Forecasting_Documentation.pdf`; [report/README.md](report/README.md) explains how to update author details and rebuild it from saved evidence. The earlier `REPORT_DRAFT.md` and `REPORT_TEMPLATE.md` remain as drafting context. Use `docs/video-outline.md` for an individual 7-10 minute presentation. Add the actual GitHub and video links after publishing them.

## Results summary (test week 16-22 Dec 2013, 1008 slots per square, all observed)

Full-period top three squares: **5161, 5059, 5259**. Evaluation squares: **5161, 4159, 4556**.

| Square | Model | MAE | MAPE % | RMSE | Final fit s | Week inference s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5161 | Ridge | 84.42 | 7.70 | 130.48 | 0.01 | 0.001 |
| 5161 | LSTM | 90.84 | 7.94 | 144.40 | 41.1 | 0.03 |
| 5161 | TCN | **83.67** | **7.52** | 133.55 | 87.5 | 0.13 |
| 4159 | Ridge | **14.17** | **6.13** | **19.43** | 0.01 | 0.001 |
| 4159 | LSTM | 14.17 | 6.24 | 19.44 | 43.4 | 0.05 |
| 4159 | TCN | 15.35 | 7.13 | 20.29 | 33.3 | 0.08 |
| 4556 | Ridge | **25.88** | **5.88** | **35.02** | 0.01 | 0.001 |
| 4556 | LSTM | 29.45 | 6.92 | 38.53 | 28.7 | 0.04 |
| 4556 | TCN | 31.51 | 7.41 | 41.90 | 76.4 | 0.15 |

These come from one seed, one test week and one CPU timing run, so treat small differences (e.g. Ridge vs LSTM on 4159) as ties. The results from tuning iteration 1 (8-epoch cap) are archived in `results/iteration1_max8_epochs/`; see `docs/methodology.md`.

## Decisions to verify with the lecturer

The brief says to forecast the highest-traffic square, then asks for **three** areas, three metric tables, and **nine** plots without identifying the other two. This implementation uses the full-period top square plus **4159 and 4556**, which the EDA section explicitly names. If either is itself top square, the three-area requirement needs a distinct replacement. If the intended three were the **top three** instead, change `evaluation_areas` in `data/processed/manifest.json` to the `top3` list before training and plotting. Preserve the actual choice in the report.

## Limits and integrity

No test metrics, top squares, pattern interpretations, runtime figures, PDF research report, video, or GitHub URL are asserted before a full data run. Missing raw square/time intervals remain unknown; past-only forward fill supports input windows, while missing target intervals are omitted from training and scoring. A missing Internet field *inside a present record* is treated as zero. These decisions should be evaluated against raw documentation and missingness rates. MAPE excludes targets equal to zero and records its denominator; it may still be unstable near zero. Models use the last 144 observed/filled intervals (24 hours); they cannot directly see a prior week. The TCN receptive field is 127 intervals, shorter than the nominal window, another issue to discuss. Results from one validation week and one test week have uncertainty and are not broad out-of-sample guarantees.
