# Bi-HYCO: Bi-Objective Hybrid-Cooperative Learning

This repository contains the implementation and numerical experiments associated with **Bi-HYCO (Bi-Objective Hybrid-Cooperative Learning)**, a multi-objective extension of the HYCO framework for combining physics-based and data-driven models.

## Overview

HYCO is a hybrid modeling strategy in which a **physical model** and a **synthetic/data-driven model** cooperate during training. Rather than incorporating the physical equations directly into the loss of a neural network, the two models are trained as separate components and exchange information through an interaction mechanism.

**Bi-HYCO** formulates this cooperation from a bi-objective optimization perspective. The two main objectives account for the quality of the physical and synthetic models, while their interaction encourages consistency between their predictions.

The framework is designed for problems involving:

* physics-based and data-driven modeling;
* partial or fragmented observations;
* inverse problems and parameter identification;
* PDE-constrained learning;
* multi-objective optimization;
* scientific machine learning.

## Main Features

* Coupling of physical and synthetic models.
* Bi-objective formulation of the HYCO framework.
* Flexible treatment of physical, synthetic, and interaction losses.
* Support for different optimization strategies.
* Numerical experiments for PDE-based applications.
* Tools for analyzing the trade-off between competing objectives and approximating Pareto solutions.

## Method

Let

* \(u_{\mathrm{phy}}\) denote the prediction of the physical model,
* \(u_{\mathrm{syn}}\) denote the prediction of the synthetic model,
* \(\mathcal{L}_{\mathrm{phy}}\) denote the physical objective,
* \(\mathcal{L}_{\mathrm{syn}}\) denote the synthetic objective, and
* \(\mathcal{L}_{\mathrm{int}}\) denote an interaction term measuring the discrepancy between the two models.

Bi-HYCO considers the cooperation between both models as a **bi-objective optimization problem**, rather than reducing the entire learning task to a single objective from the outset.

Different optimization strategies can then be used to obtain solutions representing different compromises between physical consistency and data-driven accuracy.

## Repository Structure

A typical organization of the repository is:

```text
.
├── src/                # Main implementation
├── experiments/        # Numerical experiments
├── configs/            # Configuration files and hyperparameters
├── data/               # Data used in the experiments
├── results/            # Numerical results
├── figures/            # Generated figures
├── requirements.txt    # Python dependencies
└── README.md
```

The exact structure may vary depending on the experiment.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

We recommend creating a dedicated Python environment:

```bash
python -m venv .venv
```

Activate the environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

The numerical experiments can be executed from the corresponding scripts in the `experiments/` directory.

For example:

```bash
python experiments/<experiment_name>.py
```

Specific parameters, model configurations, and training options are described in the corresponding experiment files.

## Reproducibility

The repository contains the code required to reproduce the numerical experiments presented in the associated work.

Whenever applicable, configuration files specify:

* model architecture;
* optimization parameters;
* objective weights;
* initialization;
* number of training iterations;
* numerical discretization;
* random seeds.

## Citation

If you use this code or the Bi-HYCO methodology in your research, please cite the associated paper.

```bibtex
@article{biHYCO,
  title   = {Bi-Objective Hybrid-Cooperative Learning},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

The complete citation will be updated upon publication.

## Related Work

Bi-HYCO builds upon the original **Hybrid-Cooperative Learning (HYCO)** framework, which introduces cooperative training between physics-based and synthetic models.

## License

Please see the `LICENSE` file for information about the terms of use of this repository.

## Contact

For questions, comments, or suggestions regarding the implementation, please contact the authors or open an issue in this repository.
