"""Basic util functions."""

import pickle


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

