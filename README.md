# Bi-HYCO: Bi-Objective Cooperative Learning for PDE Parameter Identification

This repository accompanies **Bi-HYCO: Bi-Objective Cooperative Learning for PDE Parameter Identification under Fragmented Observations**, by Umberto Biccari, Jun Chen, Roberto Morales, and Enrique Zuazua.

Bi-HYCO couples a numerical physical model and a neural synthetic model through agreement of their predicted states at unlabeled interaction points. Each local unit retains its own observational objective. The current manuscript studies an elliptic transmission inverse problem with observations held separately in two subdomains and with localized coefficient heterogeneity omitted from the inverse model.

- [Manuscript on arXiv](https://arxiv.org/abs/2609.06511)
- [Repository](https://github.com/DeustoTech/CoDeFeL-HYCO-Bi-objective-perspective)

This README documents the general experimental protocol and the reproducibility details referenced in Section 4 and Appendix B of the manuscript. The `BiHYCO_E0_E8_Study` package is the implementation associated with those results. Earlier notebooks are listed separately below.

**Contents:** [Reconstruction targets](#1-what-is-being-reconstructed) | [Installation and execution](#2-package-layout-and-installation) | [Experiment map](#3-experiment-map) | [Perturbations and calibration](#4-forward-problem-perturbations-and-calibration) | [Discretization and data](#5-fem-discretization-and-observations) | [Methods and selection](#6-reconstruction-methods-and-selection) | [Metrics and uncertainty](#7-error-definitions-and-uncertainty) | [Environment and timing](#8-hardware-software-precision-and-timings) | [Outputs and reuse](#9-outputs-reuse-and-independent-reruns) | [Additional material](#10-optional-diagnostics-and-earlier-material) | [Citation](#11-citation-and-funding)

## 1. What is being reconstructed?

The prescribed background coefficient is piecewise constant, with values $(a_1^\dagger,a_2^\dagger)=(2,1)$. Reference observations are generated with an additional localized Gaussian perturbation of this coefficient. The two-coefficient inverse models omit that perturbation.

The experiments therefore distinguish three reconstruction targets:

1. Recovery of the prescribed background parameters $(2,1)$.
2. Approximation of the complete heterogeneous reference coefficient.
3. Reconstruction of the reference state and prediction of held-out observations.

A reconstruction closer to the prescribed background need not minimize the physical data-fitting objective within the reduced reconstruction family. Background-parameter, complete-coefficient, state, and observation errors are reported separately.

`Bi-HYCO-PHY` and `Bi-HYCO-SYN` are the physical and synthetic outputs of the **same training run**. They are not independently trained methods. In the manuscript, `XPINN` denotes its direct piecewise neural prediction unless stated otherwise.

## 2. Package layout and installation

Keep the experiment directory intact:

```text
BiHYCO_E0_E8_Study/
  E0_Protocol_References_and_Data.ipynb
  E1_Control_Correct_Physics.ipynb
  E2_Main_Comparison_Coefficient_Mismatch.ipynb
  E3_Interaction_Ablation_and_Validation.ipynb
  E4_Robustness_Omega1_Five_Seeds.ipynb
  E5_Interaction_Weight_Sweep.ipynb
  E6_Coordinator_and_Exact_Gradient_Diagnostic.ipynb
  E7_Sensitivity_Identifiability_and_Cost.ipynb
  E8_Higher_Dimensional_Spatial_Coefficients.ipynb
  campaign_core.py
  original_elliptic_core.py
  requirements.txt
  references/       # Six archived fine-mesh reference problems
  cache/            # Observation datasets, splits, and reusable FEM data
  runs/             # Per-run configurations, checkpoints, histories, and results
  reports/          # Aggregated and paired CSV tables
  figures/          # Generated PNG and PDF figures
```

Paths below are relative to this experiment directory. If it is stored at the repository root as shown, start with:

```bash
git clone https://github.com/DeustoTech/CoDeFeL-HYCO-Bi-objective-perspective.git
cd CoDeFeL-HYCO-Bi-objective-perspective/BiHYCO_E0_E8_Study
python -m pip install -r requirements.txt
jupyter notebook
```

Use Python 3.10 or newer in an environment containing NumPy, SciPy, pandas, Matplotlib, PyTorch, Notebook, ipykernel, and nbformat. Select that same environment as the notebook kernel. If the package files are instead placed directly at repository root, launch Jupyter there, in the directory containing `campaign_core.py`.

The six files in `references/` are required inputs. `E0` validates and reuses them; it does **not** regenerate the original six fine-mesh reference solutions. Their filenames are `uref-elliptic-coeff-Omega{1,2}-m{00,01,03}.npz`. Upload or download these files together with the scripts and notebooks.

### Running the protocol

1. Run `E0` from top to bottom to validate the references and prepare or reuse the observation datasets, fixed splits, and FEM cache.
2. Run `E1` through `E5` in order. Completed compatible runs are reused across experiments.
3. Run `E7` for physical observation sensitivities and the cost summary. `E6` is an auxiliary coordinator diagnostic; it may be run before `E7` if those additional diagnostics are wanted.
4. Run `E8` only for the optional higher-dimensional extension. It is not needed for the tables in the current manuscript.

The notebook configuration cell uses:

```python
import campaign_core as C
MODE = "FULL"
C.setup(MODE)
```

`FULL` uses the prescribed scientific training budgets. `PILOT` is a short execution check, with default budgets of 8 Bi-HYCO rounds and 4 XPINN epochs for the two-coefficient problem. Pilot outputs are not manuscript results. Use a separate copy of the experiment directory for pilot checks: although run folders are separated by mode, aggregate CSV and figure filenames are shared.

Restart the kernel after changing imported implementation files so that the notebooks use the updated code. Start an independent campaign when changing the protocol; see the cache notes below.

## 3. Experiment map

Here, the region identifies the support of the omitted heterogeneity, $\delta_m$ is the relative **state** discrepancy, and $\eta$ is the observation-noise multiplier.

| Notebook | Configuration and purpose | Seeds | Relation to the current manuscript |
|---|---|---|---|
| E0 | Validate six references; prepare fixed train/validation/test observations and reusable FEM data | Prepare identifiers 1251-1255 | Reproducibility setup, Appendix B.1 |
| E1 | Correct-physics control: $\delta_m=0$, $\eta=0$; Bi-HYCO, GN-DIST, and XPINN | 1251 | Auxiliary control |
| E2 | Both perturbation locations; $\delta_m=0.03$, $\eta=0$; Bi-HYCO, GN-DIST, local/weighted GN variants, and XPINN | 1251-1253 | Sections 4.3-4.4, Tables 1-2; Appendix B.4, Table 7 |
| E3 | Both locations; compare $\lambda=0$ and $\lambda=1$ at $\delta_m=0.03$, $\eta=0$ | 1251-1253 | Section 4.5, Table 3 |
| E4 | **Omega1 only**; $(\delta_m,\eta)=(0.01,0),(0.03,0),(0.03,0.10)$ | 1251-1255 | Section 4.6, Table 4 |
| E5 | Omega1; $\delta_m=0.03$, $\eta=0$; $\lambda\in\{0,0.1,1,10\}$ | 1251-1253 | Appendix B.5, Table 8 |
| E6 | Omega1; compare one/two local steps and centralized Adam; separate deterministic coordinator diagnostic | 1251 | Auxiliary diagnostic, outside the reported table set |
| E7 | Linearized physical observation sensitivities at background $(2,1)$; collect cached cost/accuracy results | 1251 for the sensitivity cases | Section 4.7, Table 5 |
| E8 | Optional spatial coefficient with 8 or 16 parameters in Omega1 | 1251 in the supplied notebook | Additional extension, outside the current manuscript |

E3 reuses the $\lambda=1$ runs from E2. E4 reuses its first three noise-free 3% runs. E5 reuses the $\lambda=0$ and $\lambda=1$ runs. These repetitions in aggregate tables are not additional independent realizations. E4 also computes GN-weighted comparisons, although the manuscript's Table 4 presents GN-DIST and Bi-HYCO.

From an empty training cache, E1-E7 require up to 33 Bi-HYCO trainings and 7 XPINN trainings when the E5 sweep and optional E6 diagnostics are included. E7 adds no neural training. Existing compatible results can substantially reduce rerun time.

## 4. Forward problem, perturbations, and calibration

### Geometry and coefficient model

In polar coordinates, the outer boundary and internal interface are defined by

$$
R_{\mathrm{out}}(\varphi)=1+0.12\cos(3\varphi)+0.08\sin(5\varphi),
\qquad
R_{\mathrm{int}}(\varphi)=0.42\bigl(1+0.22\cos(5\varphi)\bigr).
$$

The inner subdomain is $\Omega_1$; the remaining outer shell is $\Omega_2$. Reference states solve

$$
-\nabla\cdot(a^\dagger_{\varepsilon,j}\nabla u^\dagger_{\varepsilon,j})=f
\quad\text{in }\Omega,
\qquad u^\dagger_{\varepsilon,j}=0\quad\text{on }\partial\Omega,
$$

with $f(x_1,x_2)=1+0.5\sin(\pi x_1)\cos(\pi x_2)$ and

$$
a^\dagger_{\varepsilon,j}(x)
=2\mathbf{1}_{\Omega_1}(x)+\mathbf{1}_{\Omega_2}(x)
+\varepsilon\mathbf{1}_{\Omega_j}(x)
\exp\left(-\frac{\|x-c_j\|^2}{2\sigma_j^2}\right).
$$

| Perturbed region | Gaussian center $c_j$ | Width $\sigma_j$ |
|---|---|---:|
| Omega1 | $(-0.22,-0.22)$ | 0.16 |
| Omega2 | $(0.75,0.15)$ | 0.075 |

All inverse methods know the geometry, interface, forcing, and boundary condition. Their two-coefficient physical model omits the Gaussian term. The heterogeneous diffusion in the reference FEM problem is evaluated at element centroids.

### Calibrated amplitudes

The prescribed discrepancy is

$$
\delta_m=
\frac{\|u^\dagger_{\varepsilon,j}-u^\dagger_0\|_{M_f}}
{\|u^\dagger_{\varepsilon,j}\|_{M_f}},
\qquad
\|v\|_{M_f}^2=\sum_i m_i^f v_i^2.
$$

The denominator is the norm of the **perturbed reference state**. The percentages specify a state discrepancy, not a percentage change in the diffusion coefficient.

Calibration holds the bump geometry fixed, varies the scalar amplitude, solves the fine-mesh forward problem, and matches this discrepancy to its prescribed target. The archived amplitudes are:

| Perturbed region | $\delta_m=0$ | $\delta_m=0.01$ | $\delta_m=0.03$ |
|---|---:|---:|---:|
| Omega1 | 0 | 1.5030441628878892 | 8.585084030575448 |
| Omega2 | 0 | 0.7672043967490174 | 3.180060042942005 |

These values are stored under `epsilon` in the reference NPZ files. Recomputing the discrepancy from their stored states and mass weights reproduces the targets to numerical precision. The original generator for these six archived files is not part of `E0`; a particular historical root-finding implementation or tolerance is therefore not asserted here. The optional E8 generator explicitly uses SciPy's `brentq` with `xtol=1e-10`.

To check the original six references without training any model, run this Python code from the experiment directory:

```python
from pathlib import Path
import numpy as np

for path in sorted(Path("references").glob("uref-elliptic-coeff-*.npz")):
    with np.load(path, allow_pickle=False) as ref:
        u = ref["u_reference_fine"]
        u0 = ref["u_unperturbed_fine"]
        mass = ref["lumped_mass_fine"]
        discrepancy = np.sqrt(
            np.sum(mass * (u - u0)**2) / np.sum(mass * u**2)
        )
        target = float(ref["target_model_mismatch"])
        assert np.isclose(discrepancy, target, atol=1e-10, rtol=0)
        print(path.name, float(ref["epsilon"]), discrepancy)
```

## 5. FEM discretization and observations

The forward solver uses continuous piecewise-linear finite elements on an unstructured Delaunay triangulation. The inversion mesh is generated from 900 interior, 180 outer-boundary, and 160 interface points; the reference mesh uses twice each number. Triangles are retained when their centroid is inside the outer domain and their area exceeds $10^{-8}$. Regional material assignment is determined by triangle centroids; the forcing is also evaluated there.

| Mesh | Nodes | Triangles | Use |
|---|---:|---:|---|
| Coarse | 1,240 | 2,297 | Physical inversion and state-error evaluation |
| Fine | 2,480 | 4,598 | Reference generation, discrepancy calibration, and complete-coefficient errors |

Meshes use fixed seeds: 1240 for the coarse mesh and 2239 for the fine mesh. Reference observations are interpolated linearly from the fine solution, with nearest-node fallback outside its convex hull. Physical observation predictions use barycentric interpolation of the coarse FEM state.

Each subdomain has 800 training locations, 200 validation locations, and 200 test locations in E0-E7. Training locations come from the archived references. Validation and test locations use independent, fixed sampling seeds. Methods share the locations and data realizations for each case.

For region $k$ and split $s$, observations are

$$
y_{k,i}^{s}=u^\dagger_{\varepsilon,j}(x_{k,i}^{s})
+\eta\,\sigma_{k,s}\xi_{k,i}^{s},
\qquad \xi_{k,i}^{s}\sim\mathcal N(0,1),
$$

where $\sigma_{k,s}$ is the **population** standard deviation of the clean observations in that region and split (`np.std`, `ddof=0`). The noise multiplier is applied separately for training, validation, and test observations. Independent pseudorandom streams are constructed from the case, split, region, and seed identifiers.

### What changes across seeds?

- The main neural comparisons use seeds **1251, 1252, and 1253**.
- The E4 robustness study uses **1251, 1252, 1253, 1254, and 1255**, only in Omega1.
- Bi-HYCO seeds control neural initialization and the interaction-point stream.
- XPINN seeds control neural initialization, interior collocation pools, and minibatch draws.
- At $\eta>0$, seeds additionally change Gaussian observation noise, including validation and test noise.
- Meshes, sensor locations, and initial physical coefficients remain fixed.

At $\eta=0$, observation data are identical across training seeds. GN is then deterministic and its solution is reused; repeated seed labels do not constitute independent GN realizations. At $\eta>0$, the GN solutions can vary with the observation noise.

## 6. Reconstruction methods and selection

### Bi-HYCO

Two copies begin with the same physical coefficients $(0.5,0.5)$ and the same initialized synthetic network. Each unit retains its own Adam moments. The local physical loss uses observations from Omega1; the local synthetic loss uses observations from Omega2. Both include the same state-interaction penalty.

| Quantity | E0-E7 default |
|---|---|
| Synthetic network | Two spatial inputs, four hidden layers of width 96, `tanh`, one scalar output |
| Initialization | Xavier-normal weights; zero biases |
| Communication rounds | 20,000 |
| Local Adam steps per unit | 2 |
| Initial learning rates | $5\times10^{-3}$ for physical coordinates; $8\times10^{-4}$ for network parameters |
| Learning-rate decay | Multiply both rates by 0.9995 every 100 rounds |
| Physical/synthetic alpha factors | `ALPHA_PHY=0.30`, `ALPHA_SYN=0.20` |
| Additional data-loss factors | `OMEGA_PHY=OMEGA_SYN=0.5` |
| Coordinator weights | `UNIT_WEIGHTS=(0.5,0.5)` |
| Physical regularizer | $10^{-8}\|a\|^2$, inside the physical data-loss factor |
| Synthetic regularizer | $\tfrac12 10^{-8}\|\theta\|^2$, inside the synthetic data-loss factor |
| Network-gradient clipping | Euclidean norm at most 5; physical-coordinate gradients are not clipped |
| Interaction weight | $\lambda=1$, except in the ablation and sweep |
| Validation frequency | Every 500 rounds and at the final round |

Thus the two local observation-plus-regularization terms are multiplied by **0.15** and **0.10**, respectively. These factors are distinct from both the coordinator weights and the interaction weight. In the paper, the alpha factors are denoted $a_{\mathrm{phy}}$ and $a_{\mathrm{syn}}$; they are unrelated to the diffusion coefficients $a_1,a_2$.

At each round, 1,000 new interaction points are sampled: 700 use a uniform polar angle and radial displacement in $[-0.04,0.04]$ around the interface, and 300 are uniform in area over the whole domain. The band is defined radially, not by shortest distance to the interface. Both units use the same points during their local steps.

The coordinator averages the **positive coefficients** and the network parameters separately. Averaged coefficients are mapped back to unconstrained coordinates through inverse softplus. Adam states are retained locally and are neither averaged nor reset. The synthetic network has no directly imposed boundary condition; the FEM physical model enforces the homogeneous Dirichlet condition.

### Physical Gauss-Newton baselines

All GN variants estimate both coefficients with the same full-domain FEM solver. They differ in the regional observation weights, not in the forward domain:

| Method | $(\beta_1,\beta_2)$ |
|---|---|
| GN-DIST | $(0.5,0.5)$ |
| GN-unit1 | $(1,0)$ |
| GN-unit2 | $(0,1)$ |
| GN-weighted | $\beta_1\in\{0.10,0.25,0.50,0.75,0.90\}$; $\beta_2=1-\beta_1$ |

GN-DIST aggregates local normal-equation contributions. Optimization uses logarithmic coefficient coordinates, initial coefficients $(0.5,0.5)$, bounds $[0.05,20]$, adaptive damping, no prior penalty by default, and at most 70 iterations. The implementation also stops at sufficiently small gradient or when no acceptable step remains. GN-weighted selects its candidate by the common balanced physical validation MSE.

### XPINN

XPINN uses two `tanh` subnetworks, each with four hidden layers of width 80, and shared positive diffusion coefficients initialized at $(0.5,0.5)$. A shared Adam optimizer uses learning rate $10^{-3}$ for 30,000 epochs. The loss is

$$
5(D_1+D_2)+\tfrac12(R_1+R_2)+10B+10C+2F,
$$

where $D_k$ are regional observation MSEs, $R_k$ are PDE-residual MSEs, and $B,C,F$ penalize outer-boundary values, interface state jumps, and normal-flux jumps. The unknown Gaussian perturbation does not enter the residual model.

Fixed pools contain 8,192 interior points per region, 1,024 boundary points, and 1,024 interface points. Each epoch draws with replacement 512 interior points per region and 256 points from each remaining pool, while using all training observations. Network derivatives use automatic differentiation. Interface normals use centered differences of the polar parametrization with angular increment $10^{-5}$.

### Checkpoint selection

Bi-HYCO and XPINN select the checkpoint minimizing the **balanced held-out MSE of the FEM physical state**, evaluated using the current inferred coefficients. Bi-HYCO evaluates every 500 rounds; XPINN every 1,000 epochs; both also evaluate at the final step. They complete the prescribed training budget and then restore the selected checkpoint. This is checkpoint selection, not early stopping.

For XPINN, selection therefore uses the physical FEM prediction even though the manuscript reports its direct neural state. Synthetic validation errors and background-parameter errors can be logged as diagnostics, but do not determine checkpoint selection. True background parameters, full reference coefficients, and test errors are not selection or tuning criteria.

E5 is a comparison of four prescribed interaction weights. Its post-selection comparison against the known background does not establish an observation-only tuning rule or universal optimality of a particular weight.

## 7. Error definitions and uncertainty

### Background parameters

$$
e_{\mathrm{bg}}=
\sqrt{\left(\frac{\widehat a_1-2}{2}\right)^2+(\widehat a_2-1)^2}.
$$

There is no additional division by $\sqrt{2}$.

### Complete coefficient

For fine-mesh triangle $T$ with area $|T|$, let $\widehat a_T$ and $a_T^\dagger$ denote the reconstructed and full reference coefficient values at its centroid. Then

$$
e_a=
\left(\frac{\sum_T |T|(\widehat a_T-a_T^\dagger)^2}
{\sum_T |T|(a_T^\dagger)^2}\right)^{1/2}.
$$

This area-weighted evaluation uses all 4,598 fine-mesh triangles, including the omitted heterogeneity in the reference coefficient.

### State fields

State errors use the 1,240-node inversion mesh. The reference state is interpolated from the fine mesh to these nodes. With lumped nodal weights $m_i=\sum_{T\ni i}|T|/3$,

$$
e_u(D)=
\left(\frac{\sum_{i\in I_D}m_i(\widehat u_i-u_i^\dagger)^2}
{\sum_{i\in I_D}m_i(u_i^\dagger)^2}\right)^{1/2}.
$$

For global errors, $I_D$ includes all nodes. Regional errors restrict the sums by the node's subdomain membership, using the same global lumped weights. Interface nodes are assigned to the inner mask. There is no additional equal-region reweighting in this field norm. The code adds small denominator safeguards for numerical stability.

For direct neural outputs, the network is evaluated at the coarse nodes; the same quadrature and reference are then used.

### Validation and test observations

For $s\in\{\mathrm{val},\mathrm{test}\}$,

$$
V_s(\widehat u)=\frac12\sum_{k=1}^{2}\frac{1}{N_{k,s}}
\sum_{i=1}^{N_{k,s}}
\bigl(\widehat u(x^s_{k,i})-y^s_{k,i}\bigr)^2.
$$

Each region has weight $1/2$, independently of its geometric area. These MSEs are distinct from the area-weighted field norms.

For the final saved direct-output MSE fields, the implementation first samples the neural output at the coarse mesh nodes and interpolates those nodal values to the sensors. The in-training synthetic/direct validation diagnostics instead evaluate the network at the sensor coordinates. These evaluation paths should be preserved when reproducing the saved tables.

### Across-seed summaries

Reported dispersions are **sample standard deviations** (`ddof=1`, denominator $n-1$), not population standard deviations, standard errors, or confidence intervals. Tables retain individual runs and paired comparisons where appropriate. A one-seed experiment has no estimated sample standard deviation; it should not be described as having zero uncertainty.

The population standard deviation used to scale observation noise is a separate quantity. At zero noise, repeated deterministic GN results do not provide additional independent samples. Three-seed and five-seed summaries describe the tested configurations and random components, rather than a general stability guarantee.

### Reading saved result fields

| Output to report | State-error fields | Test-MSE field |
|---|---|---|
| Bi-HYCO-PHY | `e_phy_global_error`, `e_phy_inner_error`, `e_phy_outer_error` | `test_mse` |
| Bi-HYCO-SYN | `e_direct_global_error`, `e_direct_inner_error`, `e_direct_outer_error` | `direct_test_mse` |
| GN physical state | `e_phy_*` | `test_mse` |
| XPINN direct neural state | `e_direct_*` | `direct_test_mse` |
| FEM state obtained from XPINN coefficients | `e_phy_*` | `test_mse` |

For E0-E7, `e_bg` is the background-parameter error and `e_coefficient` is the complete-coefficient error. CSV errors are stored as fractions; manuscript tables commonly display `100 * error` and `1e6 * test_MSE`. Avoid mixing the direct and physical XPINN fields when rebuilding the manuscript tables.

## 8. Hardware, software, precision, and timings

### Recorded computing environment

The reported runs were carried out on the author's Windows 11 computer with an **Intel Core Ultra 5 225H CPU**. The operating system reports approximately **31.5 GiB of usable physical RAM**. Saved notebook outputs and model storage indicate CPU execution.

| Component | Version or precision recorded for the campaign |
|---|---|
| Python notebook kernel | 3.11.15 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| PyTorch | 2.11.0+cpu |
| NumPy/SciPy FEM, calibration, GN, reference arrays, and numerical error calculations | Double precision (`float64`) |
| PyTorch networks, differentiable spectral FEM representation, and training tensors | Single precision (`float32`) |

The library versions and device are recorded by E0 in `cache/campaign_manifest.json`; the Python version is saved in notebook metadata. `requirements.txt` specifies minimum versions, not a complete environment lock. The E0 environment record is not a separate software record for every training run.

The two local units are executed sequentially in a single process. **Sequential does not mean single-threaded:** the historical BLAS/OpenMP/PyTorch thread counts were not recorded. These timings do not measure distributed parallel execution, network latency, or parallel speedup.

The implementation automatically selects CUDA when available. Check the device printed by each notebook; comparison with the reported CPU timings requires CPU execution. The recorded environment above describes the CPU campaign, not a separately validated GPU campaign.

### What the E0-E7 timers include

Timings use `time.perf_counter()` and are saved as `seconds`.

| Method | Included | Outside the measured interval |
|---|---|---|
| GN variants | Iterations, forward and sensitivity solves, damping/step search, and final forward solve | Local interpolation setup, final error evaluation, and result/history export |
| Bi-HYCO | Training, interaction sampling, local updates, aggregation, periodic validation, and in-loop checkpoint/history writes | Data/mesh/reference preparation, FEM preprocessing, model/optimizer initialization, restoration before resuming, and final best-model reload/evaluation/export |
| XPINN | Training, stochastic batches, periodic validation, and in-loop checkpoint/history writes | Pool/model/optimizer setup, restoration before resuming, and final best-model reload/evaluation/export |

**GN-weighted stores the optimization time of the selected candidate, not the total cost of the five-candidate search.** Its recorded `seconds` should not be interpreted as the full cost of weight selection.

For resumed neural runs, the recorded total combines the saved cumulative training time and the new segment. Checkpoint snapshots record elapsed time before writing that snapshot, so these accumulated values are not identical to an uninterrupted external stopwatch measurement of the entire workflow. A call returning a completed cached result also returns its stored training time, not the duration of that cache lookup.

## 9. Outputs, reuse, and independent reruns

Run directories have the form:

```text
runs/FULL/<method>/<case-and-configuration-id>/
  manifest.json
  training_checkpoint.pt    # Neural methods, when applicable
  model_best.pt            # Selected neural parameters
  training_history.csv     # Neural validation history
  result.json              # Metrics, status, and stored timing
  states.npz               # Selected coefficients and reconstructed fields
```

GN uses its corresponding history files. Not every method writes every file in this illustration. `runs/PILOT/` is used for short checks; aggregate `reports/` and `figures/` are shared between modes.

| Experiment | Main CSV outputs |
|---|---|
| E1 | `E1_control_runs.csv`, `E1_control_summary.csv` |
| E2 | `E2_main_runs.csv`, `E2_main_summary.csv`, `E2_paired_BiHYCO_minus_GN.csv` |
| E3 | `E3_ablation_runs.csv`, `E3_ablation_summary.csv`, `E3_paired_interaction.csv`, `E3_selected_vs_final.csv` |
| E4 | `E4_robustness_runs.csv`, `E4_robustness_summary.csv`, `E4_paired_background_error.csv` |
| E5 | `E5_interaction_runs.csv`, `E5_interaction_summary.csv` |
| E6 | `E6_coordinator_runs.csv`, `E6_coordinator_summary.csv` and separate deterministic histories |
| E7 | `E7_sensitivities.csv`, `E7_all_cached_runs.csv` |
| E8 | `E8_highdimensional_runs.csv`, `E8_highdimensional_summary.csv` |

`data_id` identifies the observation data used by a run; `reference_id` identifies the associated reference. Paired comparisons match the case, seed, and data identity. Different zero-noise training seeds may share a `data_id`.

E7 collects the completed results currently present in the FULL run archive. Its combined cost table and plot therefore depend on which optional runs are available, as well as on their stored timings.

Run-folder identifiers depend on the manually declared algorithm version, method, data identity, seed, and recorded parameter dictionary. They are **not** automatic hashes of all source code, software versions, or every training option. A completed compatible result is returned from cache. Neural checkpoints support continuation when their recorded configuration and data identity agree.

For the two-coefficient Bi-HYCO and XPINN routines, increasing the training budget can continue an existing checkpoint; requesting a shorter budget can return an already completed longer run. To obtain an independent rerun, use a separate copy of the campaign with the required references and no previous training results. When changing source code or unrecorded hyperparameters, use a new campaign/version rather than assuming an existing cache is invalidated automatically.

Preserve the per-run manifests, selected models, histories, observation datasets, references, and environment information together with the aggregate tables. Aggregate CSVs alone do not describe the complete execution configuration.

## 10. Optional diagnostics

### E6: coordinator diagnostic

E6 compares one versus two local Adam steps and a centralized objective for one illustrative seed. A separate deterministic gradient diagnostic tests the coordinator algebra with fixed interaction points and its own step-size schedule. It does not verify all assumptions of the convergence theorem for the practical Adam implementation.

### E8: higher-dimensional spatial coefficient

The supplied notebook uses 8 and 16 spatial parameters, Omega1 heterogeneity, $\delta_m=0.03$, $\eta=0$, seed 1251, and 120 training observations per unit. The full budgets are 1,000 Bi-HYCO rounds and 3,000 XPINN epochs. These are exploratory extension settings and are separate from the two-coefficient manuscript protocol.

The archived E8 perturbation amplitudes are **12.041707207169232** for 8 parameters and **5.944642050951994** for 16 parameters. Their baseline spatial coefficient differs from the two-constant background, so the amplitudes differ from the E0-E7 values.

E8 timings have a different scope: they include final selected-model evaluation, and E8-GN includes its regularization-grid search. Its field names and physical regularization also differ from E0-E7. Use the E8 outputs and implementation when analyzing that extension rather than merging its timings or errors into the main protocol unchanged. The current E8 manifests do not explicitly record the alpha factors, although the implementation uses the shared alpha constants.

The practical experiments use Adam, stochastic samples, and averaging of positive physical coefficients. The deterministic coordinator convergence theorem concerns a distinct iteration under stated assumptions; the numerical protocol is not a verification of those assumptions. E7 provides a local linearized physical sensitivity analysis, not a uniform identifiability result.

## 11. Citation and funding

If this repository is used in research, cite the accompanying manuscript:

```bibtex
@misc{biccari2026bihyco,
  title = {{Bi-HYCO}: Bi-Objective Cooperative Learning for {PDE} Parameter
           Identification under Fragmented Observations},
  author = {Biccari, Umberto and Chen, Jun and Morales, Roberto and Zuazua, Enrique},
  year = {2026},
  eprint = {2609.06511},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2609.06511}
}
```

The project is supported by the European Research Council through the CoDeFeL project, grant agreement **101096251**, under the European Union's Horizon Europe programme.
