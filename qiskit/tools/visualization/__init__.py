# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
# pylint: disable=invalid-name

"""Main Qiskit visualization methods."""

import sys
import warnings
from qiskit._util import _has_connection
from ._circuit_visualization import circuit_drawer, plot_circuit, generate_latex_source, \
    latex_circuit_drawer, matplotlib_circuit_drawer, _text_circuit_drawer, qx_color_scheme
from ._error import VisualizationError
from ._dag_visualization import dag_drawer
from ._matplotlib import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    from qiskit.tools.visualization._counts_visualization import plot_histogram
    from qiskit.tools.visualization._state_visualization import (plot_hinton,
                                                                 plot_bloch_vector,
                                                                 plot_state_city,
                                                                 plot_state_paulivec,
                                                                 plot_state_qsphere,
                                                                 plot_state)
else:
    warnings.warn("Matplotlib not installed.  \
                  Core visualizations not available", ImportWarning)

if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    if _has_connection('qvisualization.mybluemix.net', 443):
        from qiskit.tools.visualization.interactive import (iplot_state,
                                                            iplot_blochsphere,
                                                            iplot_cities,
                                                            iplot_qsphere,
                                                            iplot_hinton,
                                                            iplot_histogram,
                                                            iplot_paulivec)
