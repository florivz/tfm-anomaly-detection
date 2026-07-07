# Vendored Third-Party Sources

The following folders contain third-party code that is **committed directly**
into this repository (vendored, no git submodule) to keep the thesis fully
self-contained and reproducible.

## `anollm_src/` — AnoLLM
- Upstream: https://github.com/amazon-science/AnoLLM-large-language-models-for-tabular-anomaly-detection
- Vendored commit: `a051ba450743ea5a57175be305212464fb7bdc16`
- License: see `anollm_src/LICENSE`, `anollm_src/NOTICE`, `anollm_src/THIRD-PARTY-LICENSES`

## `FoMo-0D/` — FoMo-0D
- Reference: official implementation of FoMo-0D (arXiv:2409.05672)
- No git metadata was present in the local copy, so no exact commit is recorded.
- License: see `FoMo-0D/LICENSE`
- **Not included in git:** `FoMo-0D/ckpt.zip` (~32 MB pretrained checkpoint,
  excluded via `*.zip`). Obtain it from the upstream project and place it at
  `FoMo-0D/ckpt.zip` before running FoMo-0D.

Generated result files and model weights inside these folders are excluded by
`.gitignore` (repo root) and by the vendored projects' own nested `.gitignore`
files.
