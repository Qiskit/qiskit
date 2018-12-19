# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit visualization library.
"""
import warnings
from qiskit.tools.visualization import VisualizationError
from qiskit.tools.visualization._utils import _validate_input_state
from ._iplot_blochsphere import iplot_bloch_multivector
from ._iplot_cities import iplot_state_city
from ._iplot_hinton import iplot_state_hinton
from ._iplot_paulivec import iplot_state_paulivec
from ._iplot_qsphere import iplot_state_qsphere


def iplot_state(quantum_state, method='city', figsize=None):
    """Plot the quantum state.

    Args:
        quantum_state (ndarray): statevector or density matrix
                                 representation of a quantum state.
        method (str): Plotting method to use.
        figsize (tuple): Figure size in pixels.

    Raises:
        VisualizationError: if the input is not a statevector or density
        matrix, or if the state is not an multi-qubit quantum state.
    """
    warnings.warn("iplot_state is deprecated, and will be removed in \
                  the 0.9 release. Use the iplot_state_ * functions \
                  instead.",
                  DeprecationWarning)
    rho = _validate_input_state(quantum_state)
    if method == "city":
        iplot_state_city(rho, figsize=figsize)
    elif method == "paulivec":
        iplot_state_paulivec(rho, figsize=figsize)
    elif method == "qsphere":
        iplot_state_qsphere(rho, figsize=figsize)
    elif method == "bloch":
        iplot_bloch_multivector(rho, figsize=figsize)
    elif method == "hinton":
        iplot_state_hinton(rho, figsize=figsize)
    else:
        raise VisualizationError('Invalid plot state method.')
