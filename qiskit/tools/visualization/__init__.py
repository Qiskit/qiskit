# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Main QISKit visualization methods."""

import sys
from qiskit._util import _has_connection
from ._circuit_visualization import circuit_drawer, plot_circuit, generate_latex_source,\
    latex_circuit_drawer, matplotlib_circuit_drawer, qx_color_scheme
from ._error import VisualizationError
from ._state_visualization import plot_bloch_vector


if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    if _has_connection('https://qvisualization.mybluemix.net/', 443):
        from .interactive._iplot_state import iplot_state as plot_state
        from .interactive._iplot_histogram import iplot_histogram as \
            plot_histogram
    else:
        from ._state_visualization import plot_state
        from ._counts_visualization import plot_histogram

else:
    from ._state_visualization import plot_state
    from ._counts_visualization import plot_histogram
