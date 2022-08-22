#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Problem: Cooperative Localization
"""

# Import packages
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Import custom files
from data.data_generator import DataGenerator
from adapd_coop_loc import ADAPD

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

    parser = argparse.ArgumentParser(description='Testing ADAPD on Cooperative Localization problem.')

    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs to perform.')
    parser.add_argument('--num_iters', type=int, default=1500, help='Number of iterations per run.')
    parser.add_argument('--use_og', type=bool, default=True, help='Whether or not to use the One Gradient version.')

    args = parser.parse_args()

    print(f"\n")
    print(f"#" * 75)
    print(f"Running cooperative localization experiments from the ADAPD paper.")

    # Select the number of runs and maximum iterations
    n_runs = args.num_runs
    max_it = args.num_iters

    # Load data
    N = 50
    t = 5
    p = 2
    noise = 1e-2

    try:
        graph = nx.read_gpickle('data/graph.pickle')
        print('Communication graph loaded...')
    except:
        graph = nx.random_geometric_graph(N, 0.3)
        nx.write_gpickle(graph, 'data/graph.pickle')

    try:
        mixing_matrix = np.load(f'data/mixing_matrix.dat', allow_pickle=True)
        print('Mixing matrix loaded...\n')
    except:
        # Convert to adjacency matrix
        adj_mat = np.array(nx.linalg.graphmatrix.adjacency_matrix(graph).todense())

        # Get degree matrix
        degree = np.diag(np.sum(adj_mat, axis=1))

        # Generate Laplacian based mixing matrix
        mixing_matrix = np.eye(N) - (degree - adj_mat) / (np.max(degree) + 1)

        # Save information
        mixing_matrix.dump(f'data/mixing_matrix.dat')

    # Save graph layout and get positions
    agent_positions = nx.spring_layout(graph)

    # Plot communication graph structure
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    nx.draw_networkx_nodes(graph, agent_positions, nodelist=agent_positions.keys(), node_size=100,
                           node_color=(mixing_matrix != 0).sum(axis=1).tolist(),
                           vmin=min((mixing_matrix != 0).sum(axis=1).tolist()),
                           vmax=max((mixing_matrix != 0).sum(axis=1).tolist()),
                           cmap=plt.cm.Reds, edgecolors=mcolors.CSS4_COLORS['black'])
    nx.draw_networkx_edges(graph, agent_positions, edge_color=mcolors.CSS4_COLORS['black'])
    plt.show()

    # Convert to appropriate data type
    agent_pos = np.array(list(agent_positions.values()))

    # Save parameters
    eta = 2.0
    eta_inexact = 9.0
    dim_stop_tol = 1e-2
    dim_power = 4

    # Allocate space for all of the saved information
    adapd_total = np.zeros(shape=(n_runs, max_it + 1))
    adapd_dist = np.zeros(shape=(n_runs, max_it + 1))
    adapd_norm = np.zeros(shape=(n_runs, max_it + 1))
    adapd_consensus = np.zeros(shape=(n_runs, max_it + 1))

    # Same for all trials...
    starting_point = np.load('data/starting_point.dat', allow_pickle=True)

    # Loop over independent trials...
    for ind in range(n_runs):

        # Generate the data
        data_gen = DataGenerator(agent_pos, p, N, t, noise)

        # Initialize method
        params = {'eta': eta_inexact, 'stopping_tol': 1e-8, 'local_max': 100, 'lr': 0.25 * eta_inexact,
                  'dim': True, 'dim_stop_tol': dim_stop_tol, 'dim_power': dim_power, 'og': args.use_og}
        adapd_method = ADAPD(params, mixing_matrix, data_gen.agent_location, data_gen.noisy_measurement, starting_point)

        # Solve the method
        adapd_method.solve(max_it, data_gen.target_location.flatten())

        # Delete the data generation
        del data_gen

        # Save all of the information
        adapd_total[ind, :] = adapd_method.total_optimality
        adapd_dist[ind, :] = adapd_method.distance_to_opt
        adapd_norm[ind, :] = adapd_method.norm_hist
        adapd_consensus[ind, :] = adapd_method.consensus_violation

        # Delete the solver
        del adapd_method

        print(f'Run {ind+1} of {n_runs} completed.\n')

    # Plot the results
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    # Make border non-encompassing
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Plot results
    ax.plot(np.mean(adapd_total, axis=0), color=mcolors.CSS4_COLORS['dodgerblue'], alpha=1.0, lw=2.5,
            label=r'ADAPD', marker='^', markerfacecolor='none', markevery=80, markersize=10)
    # Set labels
    ax.set_xlabel('Communications', labelpad=10)
    ax.set_ylabel(r'Stationarity Violation', labelpad=10)
    plt.yscale('log')
    plt.title(f"ADAPD Performance on the Cooperative Localization Problem")
    # Set limits
    plt.show()

    # Final print statement
    print(f"#" * 75)
