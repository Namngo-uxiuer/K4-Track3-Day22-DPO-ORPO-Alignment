# Preference data

`train.parquet` and `eval.parquet` are prepared from `argilla/ultrafeedback-binarized-preferences-cleaned` with the lab schema `prompt`, `chosen`, `rejected`. See `ultrafeedback_metadata.json` for provenance.

The adapter included in this repository is explicitly labelled `CPU_FALLBACK`; the exact T4 recipe was not run on this 4 GB GPU/CPU-only PyTorch environment.
