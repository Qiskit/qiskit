# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QISKit visualization library.
"""

from ._iplot_blochsphere import iplot_blochsphere
from ._iplot_cities import iplot_cities
from ._iplot_hinton import iplot_hinton
from ._iplot_paulivec import iplot_paulivec
from ._iplot_qsphere import iplot_qsphere


def iplot_state(rho, method='city', options=None):
    """Plot the quantum state.

    Args:
        rho (ndarray): Density matrix representation
            of a quantum state vector or mized state.
        method (str): Plotting method to use.
        options (dict): Plotting settings.

    Note:
        If input is a state vector, you must first
        convert to density matrix via `qiskit.tools.qi.qi.outer`.
    """

    # Need updating to check its a matrix
    if method == "city":
        iplot_cities(rho, options)
    elif method == "paulivec":
        iplot_paulivec(rho, options)
    elif method == "qsphere":
        iplot_qsphere(rho, options)
    elif method == "bloch":
        iplot_blochsphere(rho, options)
    elif method == "hinton":
        iplot_hinton(rho, options)
    else:
        print("Unknown method '" + method + "'.")
