# Bi-HYCO: Bi-Objective Cooperative Learning for PDE Parameter Identification

This repository accompanies the paper [*Bi-HYCO: Bi-Objective Cooperative Learning for PDE Parameter Identification under Fragmented Observations*](https://arxiv.org/abs/2609.06511).
It implements Bi-Objective Hybrid-Cooperative Learning (Bi-HYCO), a framework for reconstructing PDE states and 
unknown physical parameters when a physics-based solver and a neural surrogate receive different, possibly disjoint, 
observations. The models communicate only through agreement of their predicted states at unlabeled interaction points.

![Illustrative example of bi-HYCO](bi-HYCO.png)

## Motivation

PDE-constrained inverse problems often have incomplete or fragmented observations: different sensors, regions, or data 
holders may observe different parts of the same physical system. A physics-based model contributes structure and 
interpretable parameters, while a neural model flexibly learns from data. Neither representation need be sufficient on its own.

Standard hybrid methods commonly combine all information in a single loss. Bi-HYCO instead preserves a distinct 
observational objective for each representation. This makes the tension between fitting the physical observations 
and fitting the synthetic-model observations explicit, while state interaction lets the models transfer information 
without centralizing their datasets.

## Approach implemented

Bi-HYCO defines two objectives: one for the physical model and one for the synthetic model. Both include a common 
interaction loss that compares their predicted states at unlabeled points. These interaction points provide a 
communication mechanism in the shared state space; they do not add measurements to either dataset.

The two objectives form a vector-valued optimization problem. Positive weighted scalarizations yield practical 
compromises on the Pareto frontier. In the shared-observation setting, the scalarized objective is optimized by 
alternating physical and synthetic parameter updates. Under the paper’s assumptions and with fixed interaction 
points, this deterministic alternating core has sufficient decrease, finite sequence length, and converges to a 
mixed critical point.

For fragmented observations, the implementation uses two local units that optimize their own observation-specific 
objectives and then aggregate their parameter copies through a coordinator, in the spirit of federated averaging. 
The experiments compare full Bi-HYCO with a vanishing-interaction ablation, PINN, and XPINN references. They address 
an elliptic transmission inverse problem and a nonlinear annular Navier–Stokes parameter-identification problem, 
including noise and scalarization studies.

## Repository contents

| Notebook | Purpose |
| --- | --- |
| `Experiment1.ipynb` | Elliptic transmission parameter-identification experiment with fragmented subdomain observations, interaction ablations, scalarization studies, PINN/XPINN references, and visualizations. |
| `Experiment1 - BI HYCO.ipynb` | Variant of the elliptic Bi-HYCO experiment with expanded comments and settings. |
| `Experiment2 NS.ipynb` | Annular Navier–Stokes Bi-HYCO experiment, including noise robustness and PINN/XPINN comparisons. |
| `Experiment2 BI HYCO.ipynb` | Alternative version of the Navier–Stokes Bi-HYCO workflow and output configuration. |

The notebooks create their own result directories, including `results_elliptic2/` for the elliptic experiment and a 
configured `Experiment*_outputs_*` directory for the Navier–Stokes experiments.

## Requirements

Use Python 3.10 or newer with Jupyter and the following packages:

```bash
pip install jupyter numpy scipy matplotlib pandas torch
```

The notebooks use PyTorch for the synthetic, PINN, and XPINN models. CPU execution is supported; a compatible GPU-enabled 
PyTorch installation can substantially reduce the runtime of the Navier–Stokes experiments.

## Running the simulations

From this directory, launch Jupyter:

```bash
jupyter notebook
```

Run `Experiment1.ipynb` from top to bottom for the elliptic transmission study, then run `Experiment2 NS.ipynb` for the 
Navier–Stokes study. The `BI HYCO` notebooks are alternative variants of the same respective workflows.

The Navier–Stokes notebooks default to `fast_mode: true`, which uses reduced settings for exploratory execution. 
Set `fast_mode: false` in the configuration cell to use the longer paper-style training schedule. The elliptic notebook contains noise levels, scalarization weights, mesh resolution, and training rounds in its setup cells; reduce these values for faster exploratory runs.

## Reference

U. Biccari, J. Chen, R. Morales, and E. Zuazua, *Bi-HYCO: Bi-Objective Cooperative Learning for PDE Parameter 
Identification under Fragmented Observations*, 2026. 
The manuscript is available on [arXiv:2609.06511](https://arxiv.org/abs/2609.06511).  

## Funding

This project has received funding from the European Research Council (ERC) under the European Union's Horizon Europe 
research and innovation programme (grant agreement No. 101096251, CoDeFeL).
