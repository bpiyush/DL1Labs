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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from matplotlib.pyplot import ylabel
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    predictions = predictions.argmax(axis=1)
    accuracy = np.mean(predictions == targets)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def step(model, loss_module, batch, iterator, epoch, lr=None, mode="Training"):
    """
    Performs forward/backward step for a given model on a given batch.

    Args:
        model: An instance of 'MLP', the model to train.
        loss_module: An instance of 'CrossEntropyModule', the loss module.
        batch: The current batch of data to train on.
        iterator: The iterator of the dataset.
        epoch: The current epoch.
        lr: The current learning rate.
        mode: The mode of the step, i.e. training or evaluation.
    
    Returns:
        model: The trained model (only if mode="Training").
        loss: scalar float, the average loss over the batch.
        accuracy: scalar float, the average accuracy over the batch.
    """
    x, y = batch
    x = x.reshape(x.shape[0], -1)

    # 1: forward pass
    y_pred = model.forward(x)
    loss = loss_module.forward(y_pred, y)
    accu = accuracy(y_pred, y)
    iterator.set_description(f"::::: {mode} | Epoch {epoch} | Loss: {loss:.4f} | Accu: {accu:.4f} | ")

    # 2: backpropagation (if training)
    if mode == "Training":
        assert lr is not None

        # 2b: compute loss gradients
        dout = loss_module.backward(y_pred, y)

        # 2c: backpropagate gradients
        model.backward(dout)

        # 3: update model parameters
        for layer in model.layers:
            if hasattr(layer, "params"):
                assert hasattr(layer, "grads"), \
                    f"Cannot update params for {layer} since it does not have gradients."
                for key in layer.params:
                    layer.params[key] -= lr * layer.grads[key]

        # 2a: clear gradient cache
        model.clear_cache()

    return model, loss, accu


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    avg_accuracy = []

    # evaluate model
    iterator = tqdm(data_loader, desc=f"::::: Evaluating model on given dataloader")
    for batch in iterator:
        x, y = batch
        x = x.reshape(x.shape[0], -1)

        # 1: forward pass
        y_pred = model.forward(x)
        accu = accuracy(y_pred, y)
        avg_accuracy.append(accu)

    avg_accuracy = np.mean(avg_accuracy)
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # define cosntants
    n_inputs = 32 * 32 * 3
    n_classes = 10

    # TODO: Initialize model and loss module
    model = MLP(n_inputs, hidden_dims, n_classes)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_model = {
        "model": None,
        "val_accuracy": 0,
        "epoch": None,
    }

    for epoch in range(epochs):
        train_epoch_loss = []
        train_epoch_accu = []
        val_epoch_loss = []
        val_epoch_accu = []

        # training loop
        iterator = tqdm(cifar10_loader["train"], desc=f"::::: Training | Epoch {epoch} | ")
        for batch in iterator:
            model, loss, accu = step(model, loss_module, batch, iterator, epoch, lr, mode="Training")
            train_epoch_loss.append(loss)
            train_epoch_accu.append(accu)

        # evaluate on validation set
        iterator = tqdm(cifar10_loader["validation"], desc=f"::::: Evaluating | Epoch {epoch} | ")
        for batch in iterator:
            model, loss, accu = step(model, loss_module, batch, iterator, epoch, lr, mode="Validate")
            val_epoch_loss.append(loss)
            val_epoch_accu.append(accu)
        
        train_losses.append(np.mean(train_epoch_loss))
        train_accuracies.append(np.mean(train_epoch_accu))
        val_losses.append(np.mean(val_epoch_loss))
        val_accuracies.append(np.mean(val_epoch_accu))

        print(f"::::: Finished Epoch {epoch} ")
        print(f"::::: Training | Loss: {train_losses[-1]:.4f} | Accuracy: {train_accuracies[-1]}")
        print(f"::::: Validate | Loss: {val_losses[-1]:.4f} | Accuracy: {val_accuracies[-1]}")

        # save best model
        if best_model["model"] is None or val_accuracies[-1] > best_model["val_accuracy"]:
            best_model["model"] = deepcopy(model)
            best_model["val_accuracy"] = val_accuracies[-1]
            best_model["epoch"] = epoch
            print(f"::::: Saving best model so far with validation accuracy {val_accuracies[-1]} (epoch {epoch})")

        print(f"{'- -' * 50}")

    # # TODO: Test best model
    test_accuracy = evaluate_model(best_model["model"], cifar10_loader["test"])
    print(f"::::: Test | Accuracy: {test_accuracy}")

    # # TODO: Add any information you might want to save for plotting
    logging_dict = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies,
        "best_val_accuracy": best_model["val_accuracy"],
        "best_test_accuracy": test_accuracy,
        "best_epoch": best_model["epoch"],
        "epochs": list(range(epochs)),
    }

    # set model as the best model
    model = best_model["model"]

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    from utils import plot_sequences
    
    # Plotting loss curve
    plot_sequences(
        x=logging_dict["epochs"],
        y1=logging_dict["train_loss"],
        y2=logging_dict["val_loss"],
        x_label="Epoch",
        y_label="Loss",
        y1_label="Training Loss",
        y2_label="Validation Loss",
        title=f"Loss curves for best model: MLP (NumPy) (Test accuracy: {logging_dict['best_test_accuracy']:.4f})",
        save=True,
        save_path = "results/mlp_numpy_loss.png",
        show=True,
    )

    # Plotting accuracy curve
    plot_sequences(
        x=logging_dict["epochs"],
        y1=logging_dict["train_accuracy"],
        y2=logging_dict["val_accuracy"],
        x_label="Epoch",
        y_label="Accuracy",
        y1_label="Training Accuracy",
        y2_label="Validation Accuracy",
        title=f"Accuracy curves for best model: MLP (NumPy) (Test accuracy: {logging_dict['best_test_accuracy']:.4f})",
        save=True,
        save_path = "results/mlp_numpy_accuracy.png",
        show=True,
    )
