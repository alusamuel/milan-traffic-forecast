# Fitted models

`models/models/` receives one final model file and JSON metadata file per model/area after `python run.py train`. Ridge coefficients are saved as `.npz`; PyTorch LSTM/TCN weights are saved as state dictionaries (`.pt`). JSON records the selected architecture, epoch count and log scaler statistics. The committed files come from the full-data run (`--max-epochs 40`); only load model artifacts from a trusted source.
