# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string

"""
A collection of useful quantum information functions for operators.

"""

import numpy as np


def process_fidelity(channel1, channel2):
    """Return the process fidelity between two quantum channels.

    Currently the input must be a unitary (until we decide on the channel)
    For a unitary channels the process fidelity is given by::

        F_p(U, U) = abs(Tr[ U^dagger U ])^2/d^2

    Args:
        channel1 (array_like): a quantum unitary operator.
        channel2 (array_like): a quantum unitary operator.

    Returns:
        array_like: The state fidelity F(state1, state2).
    """
    # convert input to numpy arrays
    s1 = np.array(channel1)
    s2 = np.array(channel2)

    # fidelity of two unitary vectors
    overlap = np.trace(np.dot(s1.conj().transpose(), s2))
    f_p = abs(overlap)**2 / (len(s1)**2)
    return f_p
