#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as f


### YOUR CODE HERE for part 1d

class Highway(nn.Module):
    """
    Highway network having skip-connection controlled by a dynamic gate.
    Xproj = ReLu( Wproj.Xconv_out + b_proj)
    Xgate = Sigmoid (Wgate.Xconv_out + bgate)
    """

    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.W_proj = nn.Linear(embed_size, embed_size, bias=True)
        self.W_gate = nn.Linear(embed_size, embed_size, bias=True)

    def forward(self, X_conv):
        """
        Take mini-batch of sentence of ConvNN
        @param X_conv (Tensor): Tensor with shape (max_sentence_length, batch_size, embed_size)
        @return X_highway (Tensor): combined output with shape (max_sentence_length, batch_size, embed_size)

        """
        Xproj = f.relu(self.W_proj(X_conv))
        Xgate = f.sigmoid(self.W_gate(X_conv))
        highway = torch.mul(Xgate, Xproj) + torch.mul((1 - Xgate), X_conv)
        return highway
### END YOUR CODE
