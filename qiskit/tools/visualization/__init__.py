# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
# pylint: disable=invalid-name

"""Main Qiskit visualization methods."""

import sys
from qiskit._util import _has_connection
from qiskit.tools.visualization._counts_visualization import plot_histogram
from qiskit.tools.visualization._state_visualization import (plot_state_hinton,
                                                             plot_bloch_vector,
                                                             plot_bloch_multivector,
                                                             plot_state_city,
                                                             plot_state_paulivec,
                                                             plot_state_qsphere,
                                                             plot_state)

from ._circuit_visualization import circuit_drawer, plot_circuit, generate_latex_source, \
    latex_circuit_drawer, matplotlib_circuit_drawer, _text_circuit_drawer, qx_color_scheme
from ._error import VisualizationError
from ._matplotlib import HAS_MATPLOTLIB

if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    if _has_connection('qvisualization.mybluemix.net', 443):
        from qiskit.tools.visualization.interactive import (iplot_state,
                                                            iplot_bloch_multivector,
                                                            iplot_state_city,
                                                            iplot_state_qsphere,
                                                            iplot_state_hinton,
                                                            iplot_histogram,
                                                            iplot_state_paulivec)

if HAS_MATPLOTLIB:
    from ._gate_map import plot_gate_map
