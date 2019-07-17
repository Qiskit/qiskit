# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check if number close to values of PI
"""

import numpy as np


def pi_check(inpt, eps=1e-6):
    """ Computes if a number is close to an integer
    fraction or multiple of PI.

    Args:
        inpt (float): Number to check.
        eps (float): EPS to check against.

    Returns:
        tuple: integter value if close and kind where
               kind is 'numer' if the value is in the
               numerator or 'denom' if in the denominator.
               Kind is None if input is not int frac/mult
               of PI.
    """
    if abs(inpt) < eps:
        return (0, None)
    val = inpt / np.pi
    kind = None
    if abs(val) >= 1:
        if abs(val % 1) < eps:
            val = int(val)
            kind = 'numer'

    else:
        val = np.pi / inpt
        if abs(abs(val) - abs(round(val))) < eps:
            val = int(round(val))
            kind = 'denom'

    return val, kind
