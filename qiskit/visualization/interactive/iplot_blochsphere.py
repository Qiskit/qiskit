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
Bloch sphere visualization
"""

import warnings

from qiskit.visualization.state_visualization import plot_bloch_multivector


def iplot_bloch_multivector(rho, figsize=None):
    """ Create a bloch sphere representation.
        Graphical representation of the input array, using as much bloch
        spheres as qubit are required.
        Args:
            rho (Statevector or DensityMatrix or array): An N-qubit quantum state.
            figsize (tuple): Figure size in pixels.
        Returns:
            Figure: A matplotlib figure for the visualization
        Example:
            .. code-block::

                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector
                from qiskit.visualization import iplot_bloch_multivector
                %matplotlib inline
                qc = QuantumCircuit(2)
                qc.h(0)
                qc.cx(0, 1)
                state = Statevector.from_instruction(qc)
                iplot_bloch_multivector(state)

    """
    warnings.warn(
        "The iplot_bloch_multivector function is deprecated and will be "
        "removed in a future release. The hosted code this depended on no "
        "longer exists so this is falling back to use the matplotlib "
        "visualizations. qiskit.visualization.plot_bloch_multivector should be "
        "used directly moving forward.", DeprecationWarning, stacklevel=2)
    fig = plot_bloch_multivector(rho, figsize=figsize)
    return fig
