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
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set
from utils import print_update


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*3, num_classes)
            )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
            cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model


def top1_error_rate(y_pred, y_true):
    """
    Computes the top-1 error rate.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
    Returns:
        The top-1 error rate.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    top1_error_rate = 1 - (y_pred.argmax(dim=1) == y_true).float().mean()
    #######################
    # END OF YOUR CODE    #
    #######################
    return top1_error_rate


def run_phase(model, batch, loss_modules, metric_modules, iterator, epoch, opt=None, phase="Training"):
    """
    Performs forward/backward step for a given model on a given batch.

    Args:
        model: An instance of 'MLP', the model to train.
        loss_module: An instance of 'CrossEntropyModule', the loss module.
        batch: The current batch of data to train on.
        iterator: The iterator of the dataset.
        epoch: The current epoch.
        opt: The optimizer used for training.
        phase: The phase of the step, i.e. training or evaluation.
    
    Returns:
        model: The trained model (only if mode="Training").
        loss: scalar float, the average loss over the batch.
        accuracy: scalar float, the average accuracy over the batch.
    """
    x, y = batch
    
    if phase == "Training":
        model = model.train()
    else:
        model = model.eval()

    # port data to device same as model
    x = x.to(model.device)
    y = y.to(model.device)

    # forward pass
    y_pred = model(x)
    
    # compute losses
    losses = defaultdict(float)
    for lossname, loss_module in loss_modules.items():
        losses[lossname] += loss_module(y_pred, y)
    net_loss = sum(losses.values())
    
    # compute metrics
    metrics = defaultdict(float)
    for metricname, metric_module in metric_modules.items():
        metric_value = metric_module(y_pred, y)
        metrics[metricname] = metric_value.item()

    # backpropagation (if training)
    if phase == "Training":
        assert opt is not None
        opt.zero_grad()
        net_loss.backward()
        opt.step()

    # get scalar loss values
    for lossname in losses:
        losses[lossname] = losses[lossname].item()
    net_loss = net_loss.item()

    # display update
    display = f"::::: [{phase[0].upper()}] | Epoch {epoch} | Loss: {net_loss:.3f} | "
    for metric in metrics:
        display += f"{metric}: {metrics[metric]:.3f} | "
    iterator.set_description(display)

    return model, losses, metrics


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    # Load the datasets
    train_dataset, validation_dataset = get_train_validation_set(data_dir)

    # Define the dataloaders
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_dataloader = data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Define the loss module
    loss_modules = {
        "cross_entropy": nn.CrossEntropyLoss(),
    }
    
    # Define the metric modules
    metric_modules = {
        "top1_error_rate": top1_error_rate,
    }

    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)

    # port model to device
    model = model.to(device)
    model.device = device

    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    losses = {
        "cross_entropy": defaultdict(list),
    }
    metrics = {
        "top1_error_rate": defaultdict(list),
    }
    best_model = {
        "model": None,
        "losses": {
            "cross_entropy": defaultdict(int),
        },
        "metrics": {
            "top1_error_rate": defaultdict(int),
        },
        "epoch": None,
        "monitor_metric": {
            "key": "top1_error_rate",
            "minimize": True,
        },
    }

    for epoch in range(epochs):
        epoch_losses = {
            "cross_entropy": defaultdict(list),
        }
        epoch_metrics = {
            "top1_error_rate": defaultdict(list),
        }

        # training loop
        iterator = tqdm(
            train_dataloader,
            desc=f"::::: [T] | Epoch {epoch} | ", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        )
        for batch in iterator:
            model, batch_losses, batch_metrics = run_phase(
                model, batch, loss_modules, metric_modules,
                iterator, epoch, optimizer, phase="Training",
            )
            for lossname, lossvalue in batch_losses.items():
                epoch_losses[lossname]["train"].append(lossvalue)
            for metricname, metricvalue in batch_metrics.items():
                epoch_metrics[metricname]["train"].append(metricvalue)

        for lossname, lossvalue in epoch_losses.items():
            losses[lossname]["train"].append(torch.tensor(lossvalue["train"]).mean().cpu().numpy())
        for metricname, metricvalue in epoch_metrics.items():
            metrics[metricname]["train"].append(torch.tensor(metricvalue["train"]).mean().cpu().numpy())

        # validation loop
        iterator = tqdm(
            validation_dataloader,
            desc=f"::::: [V] | Epoch {epoch} | ", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        )
        for batch in iterator:
            model, batch_losses, batch_metrics = run_phase(
                model, batch, loss_modules, metric_modules,
                iterator, epoch, optimizer, phase="Validation",
            )
            for lossname, lossvalue in batch_losses.items():
                epoch_losses[lossname]["validation"].append(lossvalue)
            for metricname, metricvalue in batch_metrics.items():
                epoch_metrics[metricname]["validation"].append(metricvalue)

        for lossname, lossvalue in epoch_losses.items():
            losses[lossname]["validation"].append(torch.tensor(lossvalue["validation"]).mean().cpu().numpy())
        for metricname, metricvalue in epoch_metrics.items():
            metrics[metricname]["validation"].append(torch.tensor(metricvalue["validation"]).mean().cpu().numpy())
        
        # update the learning rate scheduler
        scheduler.step()

        print(f"::::: Finished Epoch {epoch} ")
        print(f"::::: Training")
        train_losses = {k: v["train"][-1] for k, v in losses.items()}
        print("Losses: ", train_losses)
        train_metrics = {k: v["train"][-1] for k, v in metrics.items()}
        print("Metrics: ", train_metrics)

        print(f"::::: Validation")
        validation_losses = {k: v["validation"][-1] for k, v in losses.items()}
        print("Losses: ", validation_losses)
        validation_metrics = {k: v["validation"][-1] for k, v in metrics.items()}
        print("Metrics: ", validation_metrics)

        # save best model
        save_current = best_model["model"] is None
        monitor_metric = best_model["monitor_metric"]
        if not save_current:
            if monitor_metric["minimize"]:
                save_current = validation_metrics[monitor_metric["key"]] \
                    < best_model["metrics"][monitor_metric["key"]]
            else:
                save_current = validation_metrics[monitor_metric["key"]] \
                    > best_model["metrics"][monitor_metric["key"]]

        if save_current:
            best_model["model"] = deepcopy(model.to("cpu"))
            best_model["metrics"] = validation_metrics
            best_model["losses"] = validation_losses
            best_model["epoch"] = epoch
            print(f"::::: Saving best model so far with validation metrics {validation_metrics} (epoch {epoch})")
            os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
            torch.save(best_model, checkpoint_name)

        print(f"{'- -' * 50}")
    
    # Load best model and return it.
    best_model_dict = torch.load(checkpoint_name, map_location=torch.device("cpu"))
    model = best_model_dict["model"]

    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model = model.eval()
    model = model.to(device)
    
    # define the loss and metric modules
    loss_modules = {
        "cross_entropy": nn.CrossEntropyLoss(),
    }
    metric_modules = {
        "top1_error_rate": top1_error_rate,
    }

    # objects to collect per-batch losses and metrics
    losses = {
        "cross_entropy": defaultdict(list),
    }
    metrics = {
        "top1_error_rate": defaultdict(list),
    }

    iterator = tqdm(
        data_loader,
        desc=f"::::: [V] | Epoch 0 | ", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )
    for batch in iterator:
        model, batch_losses, batch_metrics = run_phase(
            model, batch, loss_modules, metric_modules,
            iterator, 0, opt=None, phase="Validation",
        )
        for lossname, lossvalue in batch_losses.items():
            losses[lossname]["validation"].append(lossvalue)
        for metricname, metricvalue in batch_metrics.items():
            metrics[metricname]["validation"].append(metricvalue)

    # mean across batches
    for lossname, lossvalue in losses.items():
        losses[lossname]["validation"] = torch.tensor(lossvalue["validation"]).mean().cpu().numpy()
    for metricname, metricvalue in metrics.items():
        metrics[metricname]["validation"] = torch.tensor(metricvalue["validation"]).mean().cpu().numpy()
    
    accuracy = 1 - metrics["top1_error_rate"]["validation"]
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(seed)
    test_results = {}
    
    # setup test dataset and dataloader
    print_update(":::::::::::::: Evaluating on clean set ::::::::::::::")
    test_dataset = get_test_set(data_dir, augmentation=None)
    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    # evaluate on clean test set
    accuracy = evaluate_model(model, test_dataloader, device)
    test_results["clean"] = accuracy
    
    corruption_functions = [
        gaussian_noise_transform,
        gaussian_blur_transform,
        contrast_transform,
        jpeg_transform
    ]
    severity_levels = [1, 2, 3, 4, 5]
    for corruption_function in corruption_functions:
        for severity_level in severity_levels:
            print_update(f":::::::::::::: Evaluating on {corruption_function.__name__}: {severity_level} ::::::::::::::")
            test_dataset = get_test_set(data_dir, augmentation=corruption_function(severity_level))
            test_dataloader = data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            accuracy = evaluate_model(model, test_dataloader, device)
            test_results[f"{corruption_function.__name__}_{severity_level}"] = accuracy
    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(seed)
    pass
    #######################
    # END OF YOUR CODE    #
    #######################





if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)