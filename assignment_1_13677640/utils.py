"""Basic util functions."""

import os
import pickle
import matplotlib.pyplot as plt


def load_pkl(path: str, encoding: str = "ascii"):
    """Loads a .pkl file.
    Args:
        path (str): path to the .pkl file
        encoding (str, optional): encoding to use for loading. Defaults to "ascii".
    Returns:
        Any: unpickled object
    """
    return pickle.load(open(path, "rb"), encoding=encoding)


def save_pkl(data, path: str):
    """Saves given object into .pkl file
    Args:
        data (Any): object to be saved
        path (str): path to the location to be saved at
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def plot_sequences(
        x, y1, y2, y1_label, y2_label, x_label, y_label, title,
        save=True, save_path="./results/sample.png", show=True,
    ):
    """Plots sequences y1 and y1 vs x."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x, y1, "--o", label=y1_label)
    ax.plot(x, y2, "--o", label=y2_label)
    ax.grid()
    # ax.set_title(f"Loss curves for best model: MLP (NumPy) (Test accuracy: {logging_dict['best_test_accuracy']:.4f})")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend()

    if save:
        # save_path = "results/mlp_numpy_loss.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    
    if show:
        plt.show()


