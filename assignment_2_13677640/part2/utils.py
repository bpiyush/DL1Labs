"""Helper functions"""


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

