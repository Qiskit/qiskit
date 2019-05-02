# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Qiskit visualization library.
"""
import warnings
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.utils import _validate_input_state
from .iplot_blochsphere import iplot_bloch_multivector
from .iplot_cities import iplot_state_city
from .iplot_hinton import iplot_state_hinton
from .iplot_paulivec import iplot_state_paulivec
from .iplot_qsphere import iplot_state_qsphere


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
