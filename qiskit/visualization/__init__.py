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
# pylint: disable=invalid-name

"""Main Qiskit visualization methods."""

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
from .gate_map import plot_gate_map

from .exceptions import VisualizationError
from .matplotlib import HAS_MATPLOTLIB

if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    if _has_connection('qvisualization.mybluemix.net', 443):
        from qiskit.visualization.interactive import (iplot_bloch_multivector,
                                                      iplot_state_city,
                                                      iplot_state_qsphere,
                                                      iplot_state_hinton,
                                                      iplot_histogram,
                                                      iplot_state_paulivec)
