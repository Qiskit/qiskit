# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A collection of useful functions for post processing results."""


def average_data(counts, observable):
    """Compute the mean value of an diagonal observable.

    Takes in an observable in dictionary format and then
    calculates the sum_i value(i) P(i) where value(i) is the value of
    the observable for state i.

    TODO: make the observable also include a matrix

    Args:
        counts (dict): a dict of outcomes from an experiment
        observable (dict): The observable to be averaged over. As an example
        ZZ on qubits equals {"00": 1, "11": 1, "01": -1, "10": -1}

    Returns:
        Double: Average of the observable
    """
    temp = 0
    tot = sum(counts.values())
    for key in counts:
        if key in observable:
            temp += counts[key] * observable[key] / tot
    return temp
