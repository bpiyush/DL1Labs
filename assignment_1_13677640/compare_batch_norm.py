################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import torch
import torch.nn as nn
import torch.optim as optim
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.

import matplotlib.pyplot as plt
from utils import load_pkl, save_pkl


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    default_hparams = dict(
        seed=42,
        data_dir="data/",
        batch_size=128,
        epochs=10,
        lr=0.1,
    )
    ablation_hparams = [
        {
            "hidden_dims": [128],
            "use_batch_norm": True,
        },
        {
            "hidden_dims": [128],
            "use_batch_norm": False,
        },
        {
            "hidden_dims": [256, 128],
            "use_batch_norm": True,
        },
        {
            "hidden_dims": [256, 128],
            "use_batch_norm": False,
        },
        {
            "hidden_dims": [512, 256, 128],
            "use_batch_norm": True,
        },
        {
            "hidden_dims": [512, 256, 128],
            "use_batch_norm": False,
        },
    ]

    # TODO: Run all hyperparameter configurations as requested
    results = []
    for hparams in ablation_hparams:
        hparams.update(default_hparams)
        print("-- --- --- --- --- Running experiment --- --- --- --- --- --")
        print(hparams)
        model, val_accuracies, test_accuracy, logging_info = train_mlp_pytorch.train(**hparams)
        results.append((hparams, logging_info))
        print(f"{'- -' * 50}")

    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    save_pkl(results, results_filename)

    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    results = load_pkl(results_filename)
    N = len(results)
    bn_true = [x for x in results if x[0]["use_batch_norm"]]
    bn_false = [x for x in results if not x[0]["use_batch_norm"]]

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
    for i in range(nrows):
        for j in range(ncols):

            if i:
                bn = bn_true
            else:
                bn = bn_false
        
            idx = i * ncols + j
            ax = axs[i, j]
            result = bn[j]

            hparams, logging_info = result
            ax.grid()
            if i == 1:
                ax.set_xlabel("Epochs")
            if j == 0:
                ax.set_ylabel("Accuracy")
            ax.set_ylim(0.3, 0.8)
            ax.plot(logging_info["epochs"], logging_info["train_accuracy"], "--o", label="Train")
            ax.plot(logging_info["epochs"], logging_info["val_accuracy"], "--o", label="Validation")
            ax.set_title(f"{hparams['hidden_dims']} | BatchNorm: {hparams['use_batch_norm']} | BVA: {logging_info['best_val_accuracy']:.3f}")
            ax.legend()
    
    save_path = "results/ablation_results_batch_norm.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results/ablation_results_batch_norm.pkl'
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)