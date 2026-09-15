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

```bibtex
@article{biccari2026bi,
  title={Bi-HYCO: Bi-Objective Cooperative Learning for PDE Parameter Identification under Fragmented Observations},
  author={Biccari, Umberto and Chen, Jun and Morales, Roberto and Zuazua, Enrique},
  journal={arXiv preprint arXiv:2609.06511},
  year={2026}
}
```

## Contact

For questions, comments, or suggestions regarding the implementation, please contact the authors or open an issue in this repository.
