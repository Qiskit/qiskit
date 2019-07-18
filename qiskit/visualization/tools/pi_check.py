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


def pi_check(inpt, eps=1e-6, latex=False, ndigits=5):
    """ Computes if a number is close to an integer
    fraction or multiple of PI and returns the
    corresponding string.

    Args:
        inpt (float): Number to check.
        eps (float): EPS to check against.
        latex (bool): Return latex string.
        ndigits (int): Number of digits to print
                       if returning raw inpt.

    Returns:
        str: string representation of output.
    """
    inpt = float(inpt)
    if abs(inpt) < 1e-14:
        return str(0)
    val = inpt / np.pi
    if abs(val) >= 1:
        if abs(val % 1) < eps:
            val = int(val)
            if latex:
                if val == 1:
                    str_out = r'\pi'
                elif val == -1:
                    str_out = r'-\pi'
                else:
                    str_out = r'%s\pi' % val
            else:
                if val == 1:
                    str_out = 'pi'
                elif val == -1:
                    str_out = '-pi'
                else:
                    str_out = '%spi' % val
            return str_out

    else:
        val = np.pi / inpt
        if abs(abs(val) - abs(round(val))) < eps:
            val = int(round(val))
            if val > 0:
                if latex:
                    str_out = r'\pi/%s' % val
                else:
                    str_out = 'pi/%s' % val
            else:
                if latex:
                    str_out = r'-\pi/%s' % abs(val)
                else:
                    str_out = '-pi/%s' % abs(val)
            return str_out

    # Look for all fracs in 8
    abs_val = abs(inpt)
    for numer in range(1, 9):
        for denom in range(1, 9):
            if abs(abs_val - numer / denom * np.pi) < eps:
                if inpt < 0:
                    numer *= -1
                # Found match
                if latex:
                    str_out = r'%s\pi/%s' % (numer, denom)
                else:
                    str_out = '%spi/%s' % (numer, denom)
         
                return str_out

    # nothing found
    str_out = '%.{}g'.format(ndigits) % inpt
    return str_out
