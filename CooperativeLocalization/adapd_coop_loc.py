#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Problem: Cooperative Localization
"""

# Import packages
import math
import time
import numpy as np
from models.objective_function import AugmentedLagrangian as AL
from models.objective_function import CooperativeLocalization as CL


# New ADAPD class
class ADAPD:
    '''
    Proposed class for solving decentralized nonconvex consensus problems.

    :param al_parameters: LIST containing local AL parameters (maximum iterations, stopping tolerance, penalty parameter, regularizers, etc.)
    :param agent_locs: Matrix with shape (N*T x p)
    :param agent_noises: Matrix with shape (N*T,)
    :param mixing_matrix: nxn mixing matrix (symmetric, doubly stochastic)
    :param init_guesses: Matrix with shape (N*T x p)
    '''

    def __init__(self, params, mixing_matrix, agent_locs, agent_noises, init_guesses):

        # Gather mixing matrix and number of nodes
        self.mixing_matrix = mixing_matrix
        self.num_nodes = self.mixing_matrix.shape[0]

        # Initialize data and labels
        self.agent_locs = agent_locs.copy()
        self.agent_noises = agent_noises.copy()

        # Gather appropriate parameters for local agents
        if 'eta' in params:
            self.eta = params['eta']
        else:
            self.eta = 1e-1
        if 'local_max' in params:
            self.local_max = int(params['local_max'])
        else:
            self.local_max = 10
        if 'lr' in params:
            self.lr = params['lr']
        else:
            self.lr = 1e-1
        if 'stopping_tol' in params:
            self.stopping_tol = params['stopping_tol']
        else:
            self.stopping_tol = 1e-6
        if 'og' in params:
            self.og = params['og']
        else:
            self.og = False
        if 'dim' in params:
            self.dim = params['dim']
        else:
            self.dim = False
        if 'dim_stop_tol' in params:
            self.dim_stop_tol_base = params['dim_stop_tol']
        else:
            self.dim_stop_tol_base = self.stopping_tol
        if 'dim_power' in params:
            self.dim_power = params['dim_power']
        else:
            self.dim_power = 2

        # Gather data parameters
        self.problem_dim = self.agent_locs[0].shape[0]
        self.num_targets = len(self.agent_noises[0])

        # Initialize the objective function
        self.objective_function = CL(self.num_nodes, self.problem_dim, self.num_targets)

        # Initialize the Augmented Lagrangian
        self.augmented_lagrangian = AL(self.objective_function, self.eta)

        # Primal local variables (X)
        self.local_variables = init_guesses.copy()

        # Primal consensual variables (X0)
        self.global_variables = np.zeros(shape=self.local_variables.shape)

        # Initialize the dual variables (Y, Z)
        self.local_dual_variables = np.zeros(shape=self.local_variables.shape)
        self.consensual_dual_variables = np.zeros(shape=self.local_variables.shape)
        self.old_consensual_dual_variables = np.zeros(shape=self.local_dual_variables.shape)

        # Save relevant information from each iteration
        self.consensus_violation = []
        self.norm_hist = []
        self.obj_hist = []
        self.total_optimality = []
        self.distance_to_opt = []
        self.num_grads_evaluated = 0

    def solve(self, outer_iterations, true_locations):
        '''
        Solve the decentralized problem with the proposed method

        :return:
        '''

        # Compute first violation
        cons, norm, total, dist = self.compute_violation(self.local_variables, true_locations)

        # Append
        self.consensus_violation.append(cons)
        self.norm_hist.append(norm)
        self.total_optimality.append(total)
        self.distance_to_opt.append(dist)

        # Time the algorithm
        time0 = time.time()

        # Run the outer loop (i.e. communication rounds)
        for i in range(outer_iterations):

            # Save norms, objective, accuracy, and time
            obj = 0
            num_grads_local = []

            # Check stopping tolerance is diminishing
            if self.dim:
                self.stopping_tol = self.dim_stop_tol_base / (self.num_nodes * math.pow(i + 1, self.dim_power))
            else:
                pass

            # Do local subsolvers
            for j in range(self.num_nodes):

                # Update the local solution
                if not self.og:
                    self.local_variables[j, :], num_grads = self.local_subsolver(self.local_variables[j, :].copy(),
                                                                                 self.global_variables[j, :].copy(),
                                                                                 self.local_dual_variables[j, :].copy(),
                                                                                 self.agent_locs[j, :],
                                                                                 self.agent_noises[j])

                    num_grads_local.append(num_grads)
                else:
                    self.local_variables[j, :] = self.global_variables[j, :].copy() - self.eta * (
                            self.objective_function.gradient(
                                self.agent_locs[j, :],
                                self.agent_noises[j],
                                self.local_variables[j, :].copy()).copy() + self.local_dual_variables[j, :].copy())
                    num_grads_local.append(1)

                # Compute objective value
                obj += self.objective_function.forward(
                    self.agent_locs[j, :], self.agent_noises[j], self.local_variables[j, :].copy())

            # Compute average number of gradients evaluated at this round
            avg_num_grads_i = int(sum(num_grads_local) / len(num_grads_local))
            self.num_grads_evaluated += avg_num_grads_i

            # Update X0
            self.global_variables = 0.5 * (self.local_variables.copy() + self.global_variables.copy() + self.eta * (
                    self.local_dual_variables.copy() - 2 * self.consensual_dual_variables.copy() + self.old_consensual_dual_variables.copy()
            ))

            # Update Y
            self.local_dual_variables = self.local_dual_variables.copy() + \
                                        (1 / self.eta) * (self.local_variables.copy() - self.global_variables.copy())

            # Update Z
            self.consensual_dual_variables = self.consensual_dual_variables.copy() + \
                (1 / self.eta) * np.matmul(np.eye(self.num_nodes) - self.mixing_matrix, self.global_variables.copy()).copy()

            # Compute errors
            cons, norm, total, dist = self.compute_violation(self.local_variables, true_locations)

            # Append
            self.consensus_violation.append(cons)
            self.norm_hist.append(norm)
            self.obj_hist.append(obj / self.num_nodes)
            self.total_optimality.append(total)
            self.distance_to_opt.append(dist)

        # Record time
        time1 = time.time() - time0
        print(
            f'[ADAPD] Training completed in {round(time1, 4)} seconds. Evaluated {self.num_grads_evaluated} gradients.')

    def local_subsolver(self, init_guess, current_global_variable, current_dual_variable, agent_loc, agent_noise):
        '''FISTA for local problem'''

        # Track gradient
        al_grad_norm_hist = []

        # Initialize FISTA
        t0 = 1

        # Save initial guess
        weights = np.array([init_guess], dtype=float).flatten()
        extracted_point = np.array([init_guess], dtype=float).flatten()

        # Perform for loop
        for i in range(self.local_max):
            # Format and save old weights
            old_weights = weights.copy()

            # Compute momentum
            t1 = (1 + math.sqrt(1 + 4 * t0 ** 2)) / 2
            w = (t0 - 1) / t1
            t0 = t1

            # Get AL grad
            al_grad = self.augmented_lagrangian.gradients(agent_loc, agent_noise, extracted_point, current_global_variable,
                                                          current_dual_variable)

            # Check norm
            grad_norm = np.linalg.norm(al_grad) ** 2
            if grad_norm <= self.stopping_tol:
                return weights, len(al_grad_norm_hist)

            # Update the guess
            weights = extracted_point - (self.lr) * al_grad
            extracted_point = weights + w * (weights - old_weights)

            # Change format bc numpy is particular
            extracted_point = np.array([extracted_point], dtype=float).flatten()
            weights = np.array([weights], dtype=float).flatten()

            # ONLY APPEND IF WE TAKE A LOCAL STEP
            al_grad_norm_hist.append(grad_norm)

        return weights, len(al_grad_norm_hist)

    def compute_violation(self, matrix_to_share, true_targets):

        # Allocate space for average matrix
        avg_point = (1 / self.num_nodes) * np.sum(matrix_to_share, axis=0)

        # Get consensus violation
        consensus_violation = np.linalg.norm(avg_point - matrix_to_share, ord='fro') ** 2

        # Get norm violation
        grads = 0
        for j in range(self.num_nodes):
            grads += self.objective_function.gradient(self.agent_locs[j, :], self.agent_noises[j], avg_point)

        # Compute norms
        stationarity = np.linalg.norm(grads, ord=2) ** 2

        # Save total violation
        total_violation = consensus_violation + stationarity

        # Compute distance to optimal location
        if true_targets is not None:
            distance_to_opt = np.linalg.norm(avg_point - true_targets, ord=2) ** 2
        else:
            distance_to_opt = None

        return consensus_violation, stationarity, total_violation, distance_to_opt



