#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Problem: Logistic Regression with a Non-convex Regularizer
"""

# Import packages
import argparse
import numpy as np
import sklearn.datasets as sk
import matplotlib.pyplot as plt

# Import custom files
from adapd_log_reg import ADAPD

# Change fonts
from matplotlib import rc
import matplotlib.colors as mcolors

# Style plots
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 30
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)
fd = {'size': BIGGER_SIZE, 'family': 'serif', 'serif': ['Computer Modern']}
rc('font', **fd)
rc('text', usetex=True)

# Run the file
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Testing ADAPD on Logistic Regression problem.')

    parser.add_argument('--num_iters', type=int, default=500, help='Number of iterations per run.')
    parser.add_argument('--regularization', type=float, default=1.0, help='Regularization term weight in the objective function.')
    parser.add_argument('--graph', type=str, default='ring', choices=['ring', 'random'],
                        help='Communication pattern.')

    args = parser.parse_args()

    print(f"\n")
    print(f"#" * 75)
    print(f"Running logistic regression experiments from the ADAPD paper.")

    # Load the Data
    [train_data, train_labels] = sk.load_svmlight_file('data/a9a.txt')
    [test_data, test_labels] = sk.load_svmlight_file('data/a9a_test.txt', n_features=train_data.shape[1])
    train_data = train_data.todense()
    test_data = test_data.todense()

    # Load the mixing matrix
    mixing_mat = np.load(f'data/a9a_mixing_mat_{args.graph}_50nodes.dat', allow_pickle=True)

    # Set max iters
    max_it = args.num_iters
    conv_reg_weight = args.regularization
    num_nodes = 50

    # Initialize other training parameters
    comm_rounds = 1
    adapd_step = 2.5
    stop_tol = 1e-8
    local_max = 100
    local_step = 0.5 * adapd_step

    # Initialize random starting location
    starting_point = np.random.normal(loc=0.0, scale=1.0, size=(train_data.shape[1], num_nodes))

    # Proposed Method
    adapd_method = ADAPD({'local_max_iters': local_max,
                        'local_steps': local_step, 'eta': adapd_step,
                        'beta': conv_reg_weight, 'stopping_tol': stop_tol, 'comm_rounds': 1, 'dim': True,
                        'dim_stop_tol': 1e-3, 'og': False}, mixing_mat, train_data, train_labels, starting_point)

    # Run the method
    adapd_method.solve(max_it, test_data, test_labels)

    # Plot the results
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    # Make border non-encompassing
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Plot results
    ax.plot(adapd_method.total_optimality, color=mcolors.CSS4_COLORS['dodgerblue'], alpha=1.0, lw=2.5,
            label=r'ADAPD', marker='^', markerfacecolor='none', markevery=80, markersize=10)
    # Set labels
    ax.set_xlabel('Communications', labelpad=10)
    ax.set_ylabel(r'Stationarity Violation', labelpad=10)
    plt.yscale('log')
    plt.title(f"ADAPD Performance on the Logistic Regression Problem")
    # Set limits
    plt.show()

    # Final print statement
    print(f"#" * 75)
