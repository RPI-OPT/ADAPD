#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Code provided by: Gabe Mancino-Ball
    Fully connected (3-layer) architecture for MNIST
"""

# Import packages
import torch.nn as nn


# Create custom class
class FC(nn.Module):
    def __init__(self, num_classes):
        super(FC, self).__init__()

        # Declare convolutional layers
        self.l1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.Tanh()
        )

        self.l2 = nn.Sequential(
            nn.Linear(256, 80),
            nn.Tanh()
        )

        self.l3 = nn.Sequential(
            nn.Linear(80, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        '''Forward pass of the model'''

        x = x.view(-1, 784)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x