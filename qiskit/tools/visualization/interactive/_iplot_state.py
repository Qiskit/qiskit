# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QISKit visualization library.
"""

import numpy as np
from qiskit.tools.visualization import VisualizationError
from ._iplot_blochsphere import iplot_blochsphere
from ._iplot_cities import iplot_cities
from ._iplot_hinton import iplot_hinton
from ._iplot_paulivec import iplot_paulivec
from ._iplot_qsphere import iplot_qsphere


def iplot_state(quantum_state, method='city', options=None):
    """Plot the quantum state.

    Args:
        quantum_state (ndarray): statevector or density matrix
                                 representation of a quantum state.
        method (str): Plotting method to use.
        options (dict): Plotting settings.

    Raises:
        VisualizationError: if the input is not a statevector or density
        matrix, or if the state is not an multi-qubit quantum state.
    """

    # Check if input is a statevector, and convert to density matrix
    rho = np.array(quantum_state)
    if rho.ndim == 1:
        rho = np.outer(rho, np.conj(rho))
    # Check the shape of the input is a square matrix
    shape = np.shape(rho)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise VisualizationError("Input is not a valid quantum state.")
    # Check state is an n-qubit state
    num = int(np.log2(len(rho)))
    if 2 ** num != len(rho):
        raise VisualizationError("Input is not a multi-qubit quantum state.")

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
