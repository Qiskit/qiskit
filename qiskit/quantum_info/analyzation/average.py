# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A collection of useful functions for post processing results."""

from .make_observable import make_dict_observable


def average_data(counts, observable):
    """Compute the mean value of an diagonal observable.

    Takes in a diagonal observable in dictionary, list or matrix format and then
    calculates the sum_i value(i) P(i) where value(i) is the value of the
    observable for state i.

    Args:
        counts (dict): a dict of outcomes from an experiment
        observable (dict or matrix or list): The observable to be averaged over.
        As an example, ZZ on qubits can be given as:
        * dict: {"00": 1, "11": 1, "01": -1, "10": -1}
        * matrix: [[1, 0, 0, 0], [0, -1, 0, 0, ], [0, 0, -1, 0], [0, 0, 0, 1]]
        * matrix diagonal (list): [1, -1, -1, 1]

    Returns:
        Double: Average of the observable
    """
    if not isinstance(observable, dict):
        observable = make_dict_observable(observable)
    temp = 0
    tot = sum(counts.values())
    for key in counts:
        if key in observable:
            temp += counts[key] * observable[key] / tot
    return temp
