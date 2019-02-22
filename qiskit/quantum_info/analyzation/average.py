# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A collection of useful functions for post processing results."""


def average_data(counts, observable):
    """Compute the mean value of an diagonal observable.

    Takes in a diagonal observable in dictionary, list or matrix format and then
    calculates the sum_i value(i) P(i) where value(i) is the value of the
    observable for state i.

    Args:
        counts (dict): a dict of outcomes from an experiment
        observable (dict or matrix): The observable to be averaged over. As an
        example ZZ on qubits equals {"00": 1, "11": 1, "01": -1, "10": -1}

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

def make_dict_observable(matrix_observable):
    """Convert an observable in matrix form to dictionary form.

    Takes in a diagonal observable as a matrix and converts it to a dictionary
    form. Can also handle a list sorted of the diagonal elements.

    Args:
        observable (list): The observable to be converted to dicitonary form. As

    Results:
        A dictionary with all observable states as keys, and corresponding
        values being the observed value for that state. 
    """
    raise NotImplementedError
