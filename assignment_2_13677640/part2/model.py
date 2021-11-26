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
    
    def forward_single_step(self, x, h_prev, c_prev):
        """
        Performs forward pass of a batch of inputs at a given time t.
        
        Args:
            x: input at time t.
            h_prev: previous hidden state at time t-1. Defaults to None (when t=1).
            c_prev: previous cell state at time t-1. Defaults to None (when t=1).

        Returns:
            h_next: next hidden state at time t.
            c_next: next cell state at time t.
        """
        # sanity check the shape of x
        assert len(x.shape) == 2
        assert x.shape[1] == self.embed_dim
        
        # concatenate x and h_prev
        concat = torch.cat((x, h_prev), dim=-1)

        # compute input modulation gate
        g = self.tanh(torch.mm(concat, self.W_g.T) + self.b_g)
        # compute forget gate
        f = self.sigmoid(torch.mm(concat, self.W_f.T) + self.b_f)
        # compute input gate
        i = self.sigmoid(torch.mm(concat, self.W_i.T) + self.b_i)
        # compute output gate
        o = self.sigmoid(torch.mm(concat, self.W_o.T) + self.b_o)

        # compute new cell state
        c_next = f * c_prev + i * g

        # compute next hidden state
        h_next = o * self.tanh(c_next)

        return h_next, c_next

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
        T, B, D = embeds.shape
        assert D == self.embed_dim

        h = torch.zeros(B, self.hidden_dim)
        c = torch.zeros(B, self.hidden_dim)
        h_sequence = []
        c_sequence = []
        
        for t in range(T):
            h, c = self.forward_single_step(embeds[t], h, c)
            h_sequence.append(h)
            c_sequence.append(c)

        h_sequence = torch.stack(h_sequence, dim=0)
        c_sequence = torch.stack(c_sequence, dim=0)
        
        return h_sequence, c_sequence
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
        self.__dict__.update(args)

        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.lstm_cell = LSTM(self.lstm_hidden_dim, self.embedding_size)
        self.output_layer = nn.Linear(self.lstm_hidden_dim, self.vocabulary_size)
        
        # only used while generating new sentences (since while training, we CrossEntropyLoss is used)
        self.softmax = nn.Softmax(dim=-1)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x, c_history=None, h_history=None):
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
        # x is of shape [sentence length, batch size]
        embed = self.embedding(x)
        c_history, h_history = self.lstm_cell(embed, c_history, h_history)
        # no need to project h0
        p_history = self.output_layer(h_history[1:])
        
        return p_history, c_history, h_history
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
        from tqdm import tqdm
        generated_text = torch.randint(0, self.vocabulary_size, (1, batch_size))
        
        iterator = tqdm(range(1, sample_length), desc="Generating text")
        c_history, h_history = None, None
        # import ipdb; ipdb.set_trace()
        for t in iterator:
            p, c_history, h_history = self.forward(generated_text, c_history, h_history)
            print(c_history.shape, h_history.shape)
            
            # print(c_history.shape)
            # if c_history.shape[0] > t + 1:
            #     import ipdb; ipdb.set_trace()
                

            if temperature == 0.:
                next_char = p.argmax(dim=-1)
            else:
                next_char = self.softmax(p / temperature)
                next_char = torch.multinomial(next_char, 1)
            
            generated_text = torch.cat([generated_text, next_char], dim=0)
        
        return generated_text
        #######################
        # END OF YOUR CODE    #
        #######################


if __name__ == '__main__':
    # Use this space to test your implementation and experiment.
    lstm = LSTM(10, 20)
    print(lstm)
    x = torch.randn(5, 4, 20)
    c_seq, h_seq = lstm(x)
    assert c_seq.shape == torch.Size([5, 4, 10])
    assert h_seq.shape == torch.Size([5, 4, 10])
    
    # # test text generator
    # text_generator = TextGenerationModel(
    #     args=dict(vocabulary_size=10, embedding_size=20, lstm_hidden_dim=10),
    # )
    # x = torch.randint(0, 10, (30, 4))
    # p, c, h = text_generator(x)
    # assert p.shape == torch.Size([30, 4, 10])
    
    # # generate using argmax sampling
    # generated_text = text_generator.sample(temperature=0.)
    # assert generated_text.shape == torch.Size([30, 4])

    # # generate using softmax-with-temperate sampling
    # generated_text = text_generator.sample(temperature=0.5)
    # assert generated_text.shape == torch.Size([30, 4])
    # import ipdb; ipdb.set_trace()
