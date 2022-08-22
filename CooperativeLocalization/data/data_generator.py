#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Problem: Cooperative Localization

    Data Generation
"""

# Import packages
import numpy as np


# Create custom data generation function
class DataGenerator:
    '''
    Generate data for cooperative localization problem

    :param num_samples: number of desired training/testing data points
    :param problem_dimension: dimension of the problem
    :param num_agents: number of agents in this formulation
    '''

    def __init__(self, agent_locations, problem_dimension, num_agents, num_targets, noise_level):

        # Initialize dimensions
        self.problem_dimension = problem_dimension
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.noise_level = noise_level

        # Generate agent locations
        self.agent_location = agent_locations.copy()

        # each row is a specific target
        self.target_location = np.random.normal(loc=0.0, scale=0.1, size=(self.num_targets, self.problem_dimension))

        # Get squared distance to each target, from each agent
        self.noisy_measurement = []
        for i in range(self.num_agents):

            # Generate noise
            self.noise = np.random.normal(loc=0, scale=noise_level, size=(self.num_targets, ))

            self.agent_noise = np.linalg.norm(np.array([self.agent_location[i],] * self.num_targets) - self.target_location, axis=1) ** 2 + self.noise

            self.noisy_measurement.append(self.agent_noise.tolist())



