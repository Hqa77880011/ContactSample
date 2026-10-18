# ContactSample

Implementation of contact-aware adaptive sampling with funded recovery.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Use [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) or [BIDMC](https://physionet.org/content/bidmc/1.0.0/) recording exports. A waveform CSV contains `time,ppg,hr_reference` and optional `ax,ay,az,usable_reference,connected`. Times are seconds at the recorded rate; split real acquisition gaps into separate records. The manifest has `subject,record,file,sample_rate,split`; preserve subject IDs 1–9/10–12/13–15 for the source train/development/test recipe. BIDMC patient linkage must be verified separately. The quality target uses the training reference only; online decisions never read reference pulse rate or `usable_reference`.

`fit` learns a logistic usability predictor from acquired spectral/autocorrelation features and a two-layer width-32 action-loss predictor. It fits training normalization and keeps all reference-derived targets out of replay state. The spectral/autocorrelation estimator uses a causal fourth-order 0.5–4-Hz filter, an 8-second acquired window and the 30–210-bpm grid. State carries across compatible rate epochs; sleep, probes and incompatible rate changes invalidate history. A probe updates quality age but cannot emit a fresh pulse estimate.

```bash
python -m contactsample prepare --manifest recordings.csv --config config.json
```

## Run and evaluate

```bash
python -m contactsample fit --config config.json --output outputs
python -m contactsample run --config config.json --output outputs
```

## Outputs

Preparation writes the manifest. The output folder contains `model.pt`, per-record/per-policy JSON ledgers, `metrics.json` and `power_service.png`. Energy figures use configured model costs.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
