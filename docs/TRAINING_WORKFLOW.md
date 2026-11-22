# Training and adaptation workflow

This project treats three mechanisms separately.

## 1. Context biasing

Vietnamese priority phrases are supplied to SODA as recognition context. This changes decoding preference for a session but does not change model weights. Because large bias profiles caused SODA to reject the configuration, only the first 40 personal phrases are sent.

## 2. Deterministic post-correction

User-defined heard => write rules run after final Vietnamese recognition. They are application logic, not model training. Rules must be specific and based on observed errors.

## 3. Labeled-data calibration

Corrected Vietnamese recordings are stored only after explicit consent. Dataset tooling validates format, prevents duplicates, and creates a deterministic train/holdout split. Calibration compares candidate context profiles against the active baseline and records WER/CER metrics. Promotion fails closed when the report, dataset, profile, or personal-vocabulary fingerprint has changed since evaluation.

The initial engineering gate requires at least 30 distinct labeled clips, with enough samples in both train and holdout. This minimum is not evidence of generalization; a practical personal model should use a much larger and more diverse dataset.

Run calibration from WSL in the repository root:

    PYTHONPATH=. .venv-wsl/bin/python -m training.calibrate

Promotion requires an explicit approval flag and a passing report:

    py -3 -m training.release promote --approve

Rollback is also explicit:

    py -3 -m training.release rollback --approve

## Experimental weight adaptation

Research code has reconstructed parts of the Vietnamese SODA model and changed a copy of output-layer parameters. The modified model could be loaded by native SODA, but the measured development-set accuracy did not improve. Acoustic encoder fine-tuning is therefore not part of the normal product path.

Do not describe contextual biasing or deterministic corrections as weight fine-tuning.

## Privacy

Training audio can contain sensitive speech. Keep data/ private. Do not commit or publish audio, corrected transcripts, or model artifacts without explicit authorization.
