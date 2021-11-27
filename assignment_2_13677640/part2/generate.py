"""
Script to generate text from pretrained TextGenerationModel.

Example:
>>> pyrhon generate.py --ckpt_path ./checkpoints/book_EN_grimms_fairy_tails/ckpt_epoch_best.pt.tar
"""

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
from train import set_seed
from utils import print_update, save_txt


def generate(args):
    """
    Generates text from a pretrained TextGenerationModel.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # set seed
    set_seed(args.seed)

    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    ckpt_dir = os.path.dirname(args.ckpt_path)
    ckpt_name = os.path.basename(args.ckpt_path).split(".pt.tar")[0]
    txt_file = ckpt_dir.replace("checkpoints", "assets") + ".txt"
    assert os.path.exists(txt_file), f"{txt_file} does not exist."
    dataset = TextDataset(txt_file, args.input_seq_length)
    args.vocabulary_size = len(dataset._char_to_ix)
    
    print_update(f"Generating text from model trained on {txt_file}")
    print(f"Checkpoint: {args.ckpt_path}")

    # Create model
    model = TextGenerationModel(args)
    model = model.to(args.device)
    
    # Load model checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Set model to eval mode
    model.eval()
    
    # Generate text
    generated_text_ids = model.sample(
        batch_size=args.num_samples,
        sample_length=args.input_seq_length,
        temperature=args.temperature,
    )
    generated_text = []
    for i in range(args.num_samples):
        chars = [dataset._ix_to_char[int(j)] for j in generated_text_ids[:, i]]
        sentence = "".join(chars) + "\n---\n"
        generated_text.append(sentence)
    
    # save generated text
    logs_dir = ckpt_dir.replace("checkpoints", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    save_path = os.path.join(logs_dir, f"{ckpt_name}_tau_{args.temperature}_generated_text.txt")
    save_txt(generated_text, save_path)
    print_update(f"Saved generated text to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to a model checkpoint")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Generation
    parser.add_argument('--num_samples', type=int, default=5, help='N.o. samples to be generated.')
    parser.add_argument('--temperature', type=float, default=0., help='T parameter for Softmax sampling.')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    generate(args)
