# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Testing utilities for oracle compiler."""

from inspect import getfullargspec, isfunction
from . import examples


def get_truthtable_from_function(function):
    amount_bit_input = len(getfullargspec(function).args)
    result = ""
    for decimal in range(2 ** amount_bit_input):
        entry = bin(decimal)[2:].rjust(amount_bit_input, '0')
        result += str(int(function(*[i == '1' for i in entry[::-1]])))
    return result[::-1]


def example_list():
    callables = [getattr(examples, example_name) for example_name in dir(examples)
                 if example_name[0] != '_']
    return [func for func in callables
            if isfunction(func) and 'tests/examples.py' in func.__code__.co_filename]
