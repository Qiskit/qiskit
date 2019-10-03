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
============================================
Visualizations (:mod:`qiskit.visualization`)
============================================

.. currentmodule:: qiskit.visualization

Counts and State Visualizations
===============================

.. autosummary::
   :toctree: ../stubs/

   plot_histogram - Plot a histogram of counts.
   plot_bloch_vector - Plot a Bloch vector on the Bloch sphere.
   plot_bloch_multivector - Display multiple Bloch vectors on sphere.
   plot_state_city - The cityscape of quantum state.
   plot_state_hinton - A hinton diagram for the quanum state.
   plot_state_paulivec - Plot the paulivec representation of a quantum state.
   plot_state_qsphere - The qsphere representation of a quantum state.

Interactive Visualizations
==========================

.. autosummary::
   :toctree: ../stubs/

   iplot_histogram - Interactive histogram of counts.
   iplot_bloch_multivector - Display multiple Bloch vectors on sphere.
   iplot_state_city - The cityscape of quantum state.
   iplot_state_hinton - A hinton diagram for the quanum state.
   iplot_state_paulivec - Plot the paulivec representation of a quantum state.
   iplot_state_qsphere - Interactive qsphere representation of a quantum state.

Device Visualizations
=====================

.. autosummary::
   :toctree: ../stubs/

   plot_gate_map - Display device entangling gate topology.
   plot_error_map - Plot the error rates of a given device.
   plot_circuit_layout - Display the layout of a circuit on a device.

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   VisualizationError - Base execption for visualizations.
"""

import os
import sys
from qiskit.util import _has_connection
from qiskit.visualization.counts_visualization import plot_histogram
from qiskit.visualization.state_visualization import (plot_state_hinton,
                                                      plot_bloch_vector,
                                                      plot_bloch_multivector,
                                                      plot_state_city,
                                                      plot_state_paulivec,
                                                      plot_state_qsphere)

from .pulse_visualization import pulse_drawer
from .circuit_visualization import circuit_drawer, qx_color_scheme
from .dag_visualization import dag_drawer
from .pass_manager_visualization import pass_manager_drawer
from .gate_map import plot_gate_map, plot_circuit_layout, plot_error_map

from .exceptions import VisualizationError
from .matplotlib import HAS_MATPLOTLIB

if (('ipykernel' in sys.modules) and ('spyder' not in sys.modules)) \
    or os.getenv('QISKIT_DOCS') == 'TRUE':
    if _has_connection('qvisualization.mybluemix.net', 443):
        from qiskit.visualization.interactive import (iplot_bloch_multivector,
                                                      iplot_state_city,
                                                      iplot_state_qsphere,
                                                      iplot_state_hinton,
                                                      iplot_histogram,
                                                      iplot_state_paulivec)
