"""Utility functions."""


def extract_between(string, start, end):
    """Extracts the string between two substrings.

    Arguments
    ---------
        string (str): The string to extract from.
        start (str): The substring to start extracting from.
        end (str): The substring to end extracting from.

    Returns
    -------
        str: The extracted string.
    """
    return string.split(start)[-1].split(end)[0]