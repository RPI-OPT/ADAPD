# ADAPD
---

This repository contains the implementations of the paper: _A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization_. [(Mancino-Ball, Xu, and Chen 2021)](https://arxiv.org/abs/2107.11321).

### Requirements
---
Our implementations are done in Python 3; two of the experiments (contained in the `CooperativeLocalization` and `LogisticRegression` folders) are serial implementations of ADAPD and can be performed on a generic CPU. The experiments found in the `CNNs` folder require 8 GPUs to implement.

For the CPU experiments, please install the following packages into a Python 3 environment:
```
matplotlib==3.4.2
networkx==2.4
numpy==1.19.1
scikit-learn==0.23.1
```

For the GPU experiments, `torch==1.6.0` and `mpi4py==3.0.3` are additionally required.

### Contents
---
In the paper, we tested our proposed framework on four problems: a cooperative localization problem, a logistic regression problem with a non-convex regularizer, and two image classification problems. The three folders in this repository correspond to these problems.

In the interest of space, the MNIST and CIFAR10 datasets are not uploaded here. The other two (i.e. serial) problems contain all relevant data.

- `CooperativeLocalization`
> + `data` folder containing the data for the experiments
> + `models` folder containing the objective function as a separate Python class for reference
> + `adapd_coop_loc.py` the Python implementation of the ADAPD method on the Cooperative Localization problem
> + `run_coop_loc.py` a file which re-creates the results for the ADAPD method that are plotted in the paper

- `LogisticRegression`
> + `data` folder containing the data for the experiments
> + `models` folder containing the objective function as a separate Python class for reference
> + `adapd_log_reg.py` the Python implementation of the ADAPD method on the Logistic Regression problem
> + `run_log_reg.py` a file which re-creates the results for the ADAPD method that are plotted in the paper

- `CNNs`
> + `init_weights_<...>` folder containing initial parameters for various architectures used in the experiments
> + `models` folder containing miscellaneous Python files for update model parameters and the various model architectures
> + `mixing_matrices` contains the mixing matrices used in the experiments
> + `adapd_mnist.py` Python implementation of the ADAPD method for the MNIST image classification problem
> + `adapd_cifar.py` Python implementation of the ADAPD method for the CIFAR10 image classification problem

### Usage
---

- For the seriel problems, navigate to the appropriate directory and run `run_<problem_type>.py` file.
- For the CNN problems, use `mpirun -np 8 python adapd_<dataset>.py ...` and fill in the appropriate parameters.


### Reference
---
Gabriel Mancino-Ball, Yangyang Xu, and Jie Chen. [_A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization_](https://arxiv.org/abs/2107.11321). Preprint arXiv:2107.11321, 2021.

### License
---
See the [LICENSE](LICENSE.txt) file for the license rights and limitations (MIT).

