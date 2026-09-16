# ArchCon 0.5.16 donor-safe IKEM patch

Overlay this archive at the ArchCon repository root. It contains source, cluster scripts,
documentation, and regression tests only; it contains no expression matrices, CEL archives,
prepared sweep data, or checkpoints.

Remove these obsolete root-level artifacts if they still exist after overlaying:

- `evaluate_molecular_egfr.py` (use the installed `archcon-evaluate-egfr` command or
  `scripts/evaluate_molecular_egfr.py`);
- `scripts.zip` (the maintained scripts are the flat files under `scripts/`).

## Frozen data contract

- GEO remains unchanged: 10,522 train, 585 validation, and 584 test arrays, split by connected
  source-GSE component.
- IKEM starts from 288 biopsies. All 254 measured-eGFR biopsies are held out. Four no-eGFR
  biopsies whose donor has a measured-eGFR sibling are also held out: `D006_L`, `D037_P`,
  `D106_P`, and `D118_L`.
- All 30 donor-clean no-eGFR biopsies are used for molecular pretraining: 24 train and six
  validation. The validation samples are `D076_L`, `D076_P`, `D097_L`, `D097_P`, `D184_P`,
  and `D204_L` (donors `D076`, `D097`, `D184`, and `D204`).
- The IKEM split unit is donor, stratified by the available L/P tissue pattern, with seed
  `20260915` and validation fraction `0.20`. There is no IKEM molecular test partition.

Checkpoint selection, the plateau scheduler, convergence checks, and hyperparameter selection
all use the predeclared score

```text
0.50 * GEO clean-validation MSE
+ 0.50 * donor-balanced IKEM clean-validation MSE
```

The GEO test set is evaluated only for the six frozen preprocessing-by-architecture winners.
eGFR outcomes never rank or select the encoders.

## Minimal recomputation boundary

Keep the downloaded GEO CEL archives, the canonical GSM/GSE mappings, and all 464 compatible
per-GSE raw/RMA checkpoints. Rebuild the V6 IKEM role-dependent products, the IKEM-local RMA
reference, the combined 10,522-GEO + 24-IKEM global target/probe effects, and both GEO and IKEM
global-RMA outputs. Then generate a new `archcon-pretrain-0516` prepared directory and sweep.
Do not reuse 0.5.15 model checkpoints, because their training membership and validation
selection rule differ.

See `METACENTRUM.md` for the exact resumable `qsub`, verification, sweep-generation, and final
evaluation commands.
