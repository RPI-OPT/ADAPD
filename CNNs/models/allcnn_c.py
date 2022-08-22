#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Code provided by: Gabe Mancino-Ball
    Code architecture taken from: https://arxiv.org/pdf/1412.6806.pdf
"""

# Import packages
import torch.nn as nn


# Create Architecture
class AllCNN_C(nn.Module):
    def __init__(self):
        super(AllCNN_C, self).__init__()

        num_classes = 10
        batch_norm_weight = 1e-4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), stride=1, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(96, eps=batch_norm_weight, affine=True, track_running_stats=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=1, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(96, eps=batch_norm_weight, affine=True, track_running_stats=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(192, eps=batch_norm_weight, affine=True, track_running_stats=False),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(192, eps=batch_norm_weight, affine=True, track_running_stats=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(192, eps=batch_norm_weight, affine=True, track_running_stats=False),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1), stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(192, eps=batch_norm_weight, affine=True, track_running_stats=False),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=num_classes, kernel_size=(1, 1), stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(num_classes, eps=batch_norm_weight, affine=True, track_running_stats=False),
        )

        self.avg = nn.AvgPool2d(6)

        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = self.log_soft(x.view(-1, 10))

        return x
