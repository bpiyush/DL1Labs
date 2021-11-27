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

import os
from datetime import datetime
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel


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


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # set seed
    set_seed(args.seed)
    
    # set directory to save model checkpoints
    file_name = os.path.basename(args.txt_file).split('.txt')[0]
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", file_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path_template = os.path.join(
        ckpt_dir, "ckpt_epoch_%s.pt.tar"
    )

    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=text_collate_fn,
    )
    args.vocabulary_size = len(dataset._char_to_ix)

    # Create model
    model = TextGenerationModel(args)
    model = model.to(args.device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Define loss criterion
    criterion = nn.CrossEntropyLoss(reduction="none")
    
    # To collect epoch losses
    epoch_losses = []

    # To collect best model (based on training loss)
    best_ckpt = {
        "epoch": None,
        "loss": np.inf,
    }
    
    # Training loop
    for e in range(1, args.num_epochs + 1):
        # Train model
        model.train()
        
        # To collect batch losses
        batch_losses = []
        epoch_iterator = tqdm(data_loader, desc=f"Epoch {e}")

        for batch in epoch_iterator:
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            
            # zero out parameter gradients
            optimizer.zero_grad()
            
            # model forward pass
            y_hat = model(x)
            
            # compute loss
            loss = criterion(y_hat.permute((0, 2, 1)), y)
            # sum loss across time steps and mean across examples in batch
            loss = loss.mean(dim=0).mean(dim=0)
            
            # backpropagate and update parameters
            loss.backward()
            optimizer.step()

            loss_scalar = loss.item()
            batch_losses.append(loss_scalar)
            
            # update the display
            display = f"::::: [Training] | Epoch {e} | Loss: {loss_scalar:.3f} | "
            epoch_iterator.set_description(display)

        epoch_losses.append(np.mean(batch_losses))
        
        # print summary
        print(f"::::: Finished epoch {e}. Summary:")
        print(f"::::: [Training] | Loss: {epoch_losses[-1]:.3f}")
        
        # update best model and save
        if best_ckpt["loss"] > epoch_losses[-1]:
            best_ckpt["epoch"] = e
            best_ckpt["loss"] = epoch_losses[-1]
            best_ckpt.update({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            })
            ckpt_path = ckpt_path_template % f"best"
            torch.save(best_ckpt, ckpt_path)
            print(f"::::: Loss improved to {epoch_losses[-1]:.4f}")
            print(f"::::: Saved checkpoint to {ckpt_path}")

        # Save model checkpoint
        if e % args.save_every == 0 or e in [1, args.num_epochs]:
            ckpt_path = ckpt_path_template % e
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": e,
                "loss": epoch_losses[-1],
            }
            torch.save(ckpt, ckpt_path)
            print(f"::::: Saved checkpoint to {ckpt_path}")
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')
    parser.add_argument('--save_every', type=int, default=5, help='To save ckpt after every E epochs')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    train(args)
