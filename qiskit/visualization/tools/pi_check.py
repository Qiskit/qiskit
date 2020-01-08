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
# pylint: disable=too-many-return-statements

"""Check if number close to values of PI
"""

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.exceptions import QiskitError

N, D = np.meshgrid(np.arange(1, 9), np.arange(1, 9))
FRAC_MESH = N / D * np.pi


def pi_check(inpt, eps=1e-6, output='text', ndigits=5):
    """ Computes if a number is close to an integer
    fraction or multiple of PI and returns the
    corresponding string.

    Args:
        inpt (float): Number to check.
        eps (float): EPS to check against.
        output (str): Options are 'text' (default),
                      'latex', and 'mpl'.
        ndigits (int): Number of digits to print
                       if returning raw inpt.

    Returns:
        str: string representation of output.

    Raises:
        QiskitError: if output is not a valid option.
    """
    if isinstance(inpt, ParameterExpression):
        return str(inpt)
    elif isinstance(inpt, str):
        return inpt

    def normalize(single_inpt):
        if abs(single_inpt) < 1e-14:
            return '0'
        val = single_inpt / np.pi
        if output == 'text':
            pi = 'pi'
        elif output == 'latex':
            pi = '\\pi'
        elif output == 'mpl':
            pi = '$\\pi$'
        else:
            raise QiskitError('pi_check parameter output should be text, latex, or mpl')
        if abs(val) >= 1:
            if abs(val % 1) < eps:
                val = int(round(val))
                if val == 1:
                    str_out = '{}'.format(pi)
                elif val == -1:
                    str_out = '-{}'.format(pi)
                else:
                    str_out = '{}{}'.format(val, pi)
                return str_out

        val = np.pi / single_inpt
        if abs(abs(val) - abs(round(val))) < eps:
            val = int(round(val))
            if val > 0:
                if output == 'latex':
                    str_out = '\\frac{%s}{%s}' % (pi, abs(val))
                else:
                    str_out = '{}/{}'.format(pi, val)
            else:
                if output == 'latex':
                    str_out = '\\frac{-%s}{%s}' % (pi, abs(val))
                else:
                    str_out = '-{}/{}'.format(pi, abs(val))
            return str_out

        # Look for all fracs in 8
        abs_val = abs(single_inpt)
        frac = np.where(np.abs(abs_val - FRAC_MESH) < 1e-8)
        if frac[0].shape[0]:
            numer = int(frac[1][0]) + 1
            denom = int(frac[0][0]) + 1
            if single_inpt < 0:
                numer *= -1

            if numer == 1 and denom == 1:
                str_out = '{}'.format(pi)
            elif numer == -1 and denom == 1:
                str_out = '-{}'.format(pi)
            elif numer == 1:
                if output == 'latex':
                    str_out = '\\frac{%s}{%s}' % (pi, denom)
                else:
                    str_out = '{}/{}'.format(pi, denom)
            elif numer == -1:
                if output == 'latex':
                    str_out = '\\frac{-%s}{%s}' % (pi, denom)
                else:
                    str_out = '-{}/{}'.format(pi, denom)
            elif denom == 1:
                if output == 'latex':
                    str_out = '\\frac{%s}{%s}' % (numer, pi)
                else:
                    str_out = '{}/{}'.format(numer, pi)
            else:
                if output == 'latex':
                    str_out = '\\frac{%s%s}{%s}' % (numer, pi, denom)
                else:
                    str_out = '{}{}/{}'.format(numer, pi, denom)

            return str_out
        # nothing found
        str_out = '%.{}g'.format(ndigits) % single_inpt
        return str_out

    complex_inpt = complex(inpt)
    real, imag = map(normalize, [complex_inpt.real, complex_inpt.imag])

    if real == '0' and imag != '0':
        return imag + 'j'
    elif real != 0 and imag != '0':
        return '{}+{}j'.format(real, imag)
    return real
