#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e

import torch
import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    """
    1D convolution layer
    """

    def __init__(self, embed_size=50, max_word_len=21, kernel_size=5, f=50, stride=1):
        super(CNN, self).__init__()
        self.cnnlayer = nn.Conv1d(in_channels=embed_size, kernel_size=kernel_size,
                                  out_channels=f, stride=stride)
        self.maxpool = nn.MaxPool1d(kernel_size=max_word_len - kernel_size + 1)

    def forward(self,X_reshaped):
        """
        map from X_reshaped to X_conv_out
        """
        X_conv = self.cnnlayer(X_reshaped)
        x_conv_out = self.maxpool(f.relu(X_conv))
        return x_conv_out.squeeze(-1)
### END YOUR CODE
