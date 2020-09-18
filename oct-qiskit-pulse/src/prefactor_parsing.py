import re
from typing import List, Tuple, Mapping, Union


def is_int(s: str) -> bool:
    """Test if string is int

    Args:
        s (str): Input String

    Returns:
        bool: result of test 
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def pull_control(tokens: List[str], variables: Mapping[str, int]) -> Tuple[List[str], str]:
    """If token represents a channel, separate the type of channel and the index
    of said channel.

    Args: tokens (List[str]): Input list of tokens, potentially a channel and a
        control or drive notation. variables (List[str]): A list of potential
        variables to match the tokens to.

    Returns: Tuple[List[str], str]: If the tokens represent a channel, separate
        the channel from the index and return the separated options, otherwise
        return None instead of the index.
    """

    # ! assert that if there is drive or control term it is second token
    if len(tokens) == 1:
        return tokens, None
    assert (tokens[0] in variables)
    if tokens[1] in variables:
        return tokens, None
    else:
        return [tokens[0]], tokens[1]

    pass


def prefactor_parser(equation: str, variables: Mapping[str, int]) -> Tuple[Union[str, int], int]:
    """Parse an equation, if it contains a variable replace it with the numeric value of the variable, if instead we recieve a channel, return the channel.

    Args:
        equation (str): The input equation, usually in the form of 'wq0/2' or 'D0'
        variables (Mapping[str, int]): A dictionary of the numeric values of each variable.

    Returns:
        Tuple[Union[str, int], int]: The numeric value of the equation or the channel information.
    """
    # Need to support parsing for equation with * and / and variables that may contain numbes
    # We are going to assume it is always two numbers o

    negative = 1
    if equation[0] == '-':
        equation = equation[1:]
        negative = -1

    tokens = equation.split('*')
    eq_type = 'mult'
    control = None
    if len(tokens) == 1:
        tokens = re.split('\/', equation)
        eq_type = 'div'
    else:
        tokens, control = pull_control(tokens, variables)
    # print(tokens)
    if len(tokens) == 1:
        return variables[tokens[0]], control
    for i, token in enumerate(tokens):
        if is_int(token):
            tokens[i] = int(token)
        else:
            tokens[i] = variables[token]
    if control:
        return (variables[tokens[0]], control)
    if eq_type == 'mult':
        return negative * tokens[0] * tokens[1], None
    else:
        return negative * tokens[0] / tokens[1], None
