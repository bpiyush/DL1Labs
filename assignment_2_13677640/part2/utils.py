"""Helper functions"""
import os
import pickle
import matplotlib.pyplot as plt
import json


def print_update(message: str, width: int = 120, fillchar: str = ":") -> str:
    """Prints an update message
    Args:
        message (str): message
        width (int): width of new update message
        fillchar (str): character to be filled to L and R of message
    Returns:
        str: print-ready update message
    """
    message = message.center(len(message) + 2, " ")
    print(message.center(width, fillchar))


def save_txt(data: list, path: str):
    """Writes data (lines) to a txt file.

    Args:
        data (list): List of strings
        path (str): path to .txt file
    """
    assert isinstance(data, list)

    lines = "\n".join(data)
    with open(path, "w") as f:
        f.write(str(lines))


def plot_sequences(
        x, y1, y2, y1_label, y2_label, x_label, y_label, title,
        save=True, save_path="./results/sample.png", show=True,
    ):
    """Plots sequences y1 and y1 vs x."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if len(y1):
        ax.plot(x, y1, "--o", label=y1_label)
    if len(y2):
        ax.plot(x, y2, "--o", label=y2_label)
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    
    if show:
        plt.show()

