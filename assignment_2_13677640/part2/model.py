###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Adapted: 2021-11-11
###############################################################################

import math
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """
    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.W_g = nn.Parameter(torch.zeros(self.hidden_dim, (self.hidden_dim + self.embed_dim)), requires_grad=True)
        self.W_i = nn.Parameter(torch.zeros(self.hidden_dim, (self.hidden_dim + self.embed_dim)), requires_grad=True)
        self.W_f = nn.Parameter(torch.zeros(self.hidden_dim, (self.hidden_dim + self.embed_dim)), requires_grad=True)
        self.W_o = nn.Parameter(torch.zeros(self.hidden_dim, (self.hidden_dim + self.embed_dim)), requires_grad=True)
        
        self.b_g = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # initialize weight matrices
        for W in [self.W_g, self.W_i, self.W_f, self.W_o]:
            nn.init.uniform_(W, -(1.0 / math.sqrt(self.hidden_dim)), (1.0 / math.sqrt(self.hidden_dim)))
        
        # initialize bias for forget gate
        nn.init.constant_(self.b_f, 1.0)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        T, B, _ = embeds.shape
        
        c_out, h_out = [torch.zeros(B, self.hidden_dim)] , [torch.zeros(B, self.hidden_dim)]
        
        for t in range(1, T):
            x_t = embeds[t]
            c_prev, h_prev = c_out[-1], h_out[-1]
            
            g = self.tanh(torch.matmul(torch.cat([x_t, h_prev], dim=1), self.W_g.T) + self.b_g)
            i = self.sigmoid(torch.matmul(torch.cat([x_t, h_prev], dim=1), self.W_i.T) + self.b_i)
            f = self.sigmoid(torch.matmul(torch.cat([x_t, h_prev], dim=1), self.W_f.T) + self.b_f)
            o = self.sigmoid(torch.matmul(torch.cat([x_t, h_prev], dim=1), self.W_o.T) + self.b_o)
            
            c_t = f * c_prev + i * g
            h_t = o * self.tanh(c_t)
            
            c_out.append(c_t)
            h_out.append(h_t)

        c_out = torch.stack(c_out, dim=0)
        h_out = torch.stack(h_out, dim=0)

        return h_out
        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """
    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


if __name__ == '__main__':
    # Use this space to test your implementation and experiment.
    lstm = LSTM(10, 20)
    x = torch.randn(5, 4, 20)
    h = lstm(x)
    assert h.shape == torch.Size([5, 4, 10])
