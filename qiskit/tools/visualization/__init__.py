# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Main QISKit visualization methods."""

import sys

from ._circuit_visualization import circuit_drawer, plot_circuit, generate_latex_source,\
    latex_circuit_drawer, matplotlib_circuit_drawer, qx_color_scheme
from ._state_visualization import plot_bloch_vector
from ._counts_visualization import plot_histogram

if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    import requests
    if requests.get(
            'https://qvisualization.mybluemix.net/').status_code == 200:
        from .interactive._iplot_state import iplot_state as plot_state
    else:
        from ._state_visualization import plot_state
else:
    from ._state_visualization import plot_state
