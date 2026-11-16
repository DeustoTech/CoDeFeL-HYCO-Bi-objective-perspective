# Allen–Cahn: extend three completed seeds to five

## Purpose

Reuse completed **seeds 7, 11 and 23**, and compute only **seeds 37 and 53** for
GN-DIST, joint-Adam Bi-HYCO and XPINN-C. These are five paired
observation/training realizations of the SAME noiseless, two-unknown Allen–Cahn
problem. Diffusion `nu` and background reaction `alpha` are the only unknowns.
The four forcing amplitudes are known; the three omitted reaction heterogeneities
and calibrated state discrepancy of approximately 0.12 are unchanged.

This package does not retune any model, optimizer, loss weight, sampler, stopping
rule or checkpoint criterion. It is not the earlier six-parameter experiment.

## Installation

Extract **all** ZIP contents directly into the working folder used by the prior
notebooks. Keep the following new files together:

```text
working_folder/
    AllenCahn_Complete_FiveSeeds.ipynb
    AllenCahn_Complete_FiveSeeds.py
    README_Complete_FiveSeeds.md
    _five_seed_support/
        Frozen_Config.json
        AllenCahn_TwoParameter.py
    AllenCahn_TwoParameter.py             # your existing numerical module, unchanged
    AllenCahn_TwoParameter_Results/       # existing reference and completed runs
        pilot/
            reference_cache/
            runs/seed7_.../
            runs/seed11_.../
            runs/seed23_.../
            reports/seed7_.../
            reports/seed11_.../
            reports/seed23_.../
            three_seed_campaign_7_11_23/ # left unchanged, when present
```

The uniquely named `_five_seed_support/` directory avoids overwriting the older
three-seed package. Its numerical source is a byte-identical backup of the original
module. The runner prefers your existing `AllenCahn_TwoParameter.py` and checks its
normalized source hash. The backup is used only when that existing module is absent.
A changed module is not silently replaced with the backup.

Use your original Python/Jupyter environment; this extension introduces no new
Python dependencies. NumPy, SciPy, pandas, PyTorch and matplotlib are already
required by the original experiment. Checkpoints are trusted local files; do not
substitute arbitrary third-party `.pt` files.

## Execution

Open the new notebook and leave:

```python
MODE = "pilot"
ROOT = "AllenCahn_TwoParameter_Results"
NEW_SEEDS = (37, 53)
RUN_SENSITIVITY = True
```

Run the cells in order: setup, preflight, seed 37, seed 53, aggregate. Computations
are sequential, not concurrent. `ROOT` can instead be an absolute path to the
existing results directory; it must name the directory *above* `pilot/`.

The existing three runs must already be complete, including their per-seed
reports, saved models and observations. A notebook file alone does not contain
those trajectories and checkpoints. The runner stops BEFORE new pilot training
if existing files are missing, incomplete or incompatible. Finish any missing
exports with the previous notebook rather than deleting existing results.

Do not execute this notebook and an older training/reporting notebook against the
same run folders at the same time.

## Exactly what is reused

- The heterogeneous and homogeneous references, calibration and refinement check.
- Existing models, metrics, timing, histories, state trajectories and sensitivities
  for seeds 7, 11, 23, read-only.
- A new seed's own checkpoint if that new training was interrupted.
- Completed seed-37/53 methods and final exports on subsequent runs.

For each NEW seed, the code generates independent observation locations, neural
initializations and sampling streams. All three methods receive the same data
within that seed. New networks start from their prescribed random initialization,
**never** from a network trained for an existing seed. Physical initial values
remain `(0.055, 50)`. No additional observational noise is introduced.

## Unchanged optimization protocol

| Setting | Bi-HYCO | XPINN-C |
|---|---:|---:|
| Maximum budget | 3,000 communication rounds | 10,000 epochs |
| Minimum before early stopping | 1,500 rounds | 5,000 epochs |
| Patience | 400 rounds | 2,000 epochs |
| Validation interval | 20 rounds | 200 epochs |
| Physical learning rate | 0.001 | 0.0002 |
| Neural learning rate | 0.0008 | 0.0008 |

Bi-HYCO uses two joint Adam updates per unit per round, persistent local Adam
moments, and equal-weight coordinator averaging of both parameter blocks. The
physical coefficients are averaged in their original coordinates. The number of
interaction points is 640, interaction weight 0.18, ramp 150 rounds, cooperative
initial-condition weight 0.1, and pretraining 1,000 steps with its separate IC
weight 1.5.

XPINN-C retains its four networks, 900-epoch warm-up, 900-epoch residual ramp,
50/25/25 mixed collocation sampling, all loss weights and all batch sizes.
GN-DIST retains its 60-iteration cap, damping, line search and numerical stopping
tests. For neural methods, a 0.3% relative gain controls patience only; exact
best validation checkpoints are stored separately for physical and neural scores.

The frozen JSON includes all settings, not only those listed in this README.
No `force_continue` override is enabled. An early-stopped method counts as a
completed realization under the same original protocol.

## Resuming safely

Rerun the notebook setup/preflight and the unfinished new-seed cell. Parameters,
optimizer states, random-number states and checkpoint selections are restored
from that method and seed. Completed methods are not retrained. If the export
was interrupted after training finished, the final training checkpoint is reused
to regenerate the export, without repeating training.

Neither `run_seed(7)`, `run_seed(11)` nor `run_seed(23)` is allowed by this runner.
Integrity checks cover the old configurations, metrics, selected and last
checkpoints, histories, observation arrays and shared reference. The earlier
three-seed aggregate is left untouched. Large existing state/figure files are
never opened for writing; the integrity manifest explicitly records the files
that are hashed.

## Output directories

New runs and per-seed reports use the original layout:

```text
AllenCahn_TwoParameter_Results/pilot/
    runs/seed37_<data_id>/<method>_<configuration_hash>/
    runs/seed53_<data_id>/<method>_<configuration_hash>/
    reports/seed37_<data_id>/
    reports/seed53_<data_id>/
    five_seed_campaign_7_11_23_37_53/
```

The combined report contains:

- `metrics_per_seed.csv`: all **15** method–seed results.
- `mean_std_long.csv`: unrounded mean, sample SD, minimum and maximum per metric.
- `parameter_mean_std.csv` and `parameters_five_seeds.tex`.
- `state_cost_mean_std.csv` and `state_and_cost_five_seeds.tex`.
- `Tables_Standalone.tex`, compilable with `amsmath` and `booktabs`.
- `training_checkpoints_and_costs.csv`: selected checkpoints, stopping iterations
  and computational costs.
- `figures/`: five individual-seed markers plus mean/sample SD, 600 dpi, 16-point
  labels, no titles.
- `sensitivity/`: all available per-seed sensitivities, discrepancy projections,
  directional checks and profiled interaction diagnostics. Vector directions are
  kept per seed rather than averaged without controlling their arbitrary signs.
- `supplementary_gn_unit1_per_seed.csv`: the original local-data control, when run;
  it is not a fourth principal method.
- `campaign_configuration.json` and `report_manifest.json`: provenance, hashes,
  data IDs, completeness and interpretation.

Each new run also stores full model states, Adam states, final and selected
checkpoints, training histories, physical/neural state trajectories, tables,
comparison figures and 12 snapshots, as in the original module. Figures remain
600 dpi with 16-point text and time annotations, and are exported in PNG/PDF.

Set `RUN_SENSITIVITY=False` only to postpone the new post-training analyses.
This does not alter training. Repeating the new-seed cell with `True` computes
missing analyses without repeating completed optimization. Coverage is reported;
no missing sensitivity output is described as computed.

## Statistical conventions

Five completed seeds are required for every principal method. The runner refuses
to silently exclude failures or report means from different seed counts. It averages
per-run errors, not parameter estimates followed by one error calculation.
Sample SD uses **`ddof=1`**, hence denominator **4**. CSV errors are fractions;
publication tables multiply relative errors by 100.

Parameter estimates use physical-validation checkpoints. Bi-HYCO physical and
neural outputs may use different checkpoints. Only the direct neural state of
XPINN-C appears in the main state table. The paired neural prediction at the
physical checkpoint remains available as a diagnostic. Bi-HYCO has one training
cost, not one separate charge per output. Timing follows the original module,
excluding reference preparation, final post-processing and separately timed
fixed-gradient/sensitivity diagnostics.

The physical problem does not change across seeds. Variation includes observation
sampling and neural training randomness. Seed 7 and the problem design have already
been used for development; the combined report is a descriptive robustness study,
not an independent blind evaluation. Standard deviations are not confidence intervals.

## Functionality test

`MODE="smoke"` uses a completely separate `ROOT + "_SMOKE_ONLY"` directory. It
creates tiny baseline realizations 7/11/23 if they do not exist, then tests adding
37/53. This tests execution and reporting only, NOT scientific accuracy. It never
uses pilot results as smoke results. Actual validation checks performed for this
package are recorded in `VALIDATION_REPORT.json`.

## Optional command-line execution

```bash
python AllenCahn_Complete_FiveSeeds.py --root AllenCahn_TwoParameter_Results
```

The Jupyter interface is recommended for inspecting the preflight and each seed
before the final aggregation.
