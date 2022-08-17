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

Install visualization optionals
-------------------------------
`pip install qiskit[visualization]`

Common parameters
-----------------

The figures created by counts and state visulizations functions (listed in the table below) are
genereted by `Matplotlib`. Some of the common parameters are listed here:

- title (str): a text string to use for the plot title
- legend (list): a list of strings to use for labels of the data.
- figsize (tuple): figure size in inches 
- color (str or list): string or lists of strings for plotting
- ax (Matplotlib.axes.Axes): An optional Axes object to be used for the visualization output. If
  none is specified a new matplotlib Figure will be created and used. Additionally, if specified
  there will be no returned Figure since it is redundant.
- filename (str) – file path to save image to.

Use title and legend
--------------------

Change fig size
---------------

Change color
------------

Reuse axes
----------

Save figure to file
-------------------


.. autosummary::
   :toctree: ../stubs/

   plot_histogram plot_bloch_vector plot_bloch_multivector plot_state_city plot_state_hinton
   plot_state_paulivec plot_state_qsphere

Device Visualizations
=====================

.. autosummary::
   :toctree: ../stubs/

   plot_gate_map plot_error_map plot_circuit_layout plot_coupling_map

Circuit Visualizations
======================

.. autosummary::
   :toctree: ../stubs/

   circuit_drawer ~qiskit.visualization.qcstyle.DefaultStyle

DAG Visualizations
==================

.. autosummary::
   :toctree: ../stubs/

   dag_drawer

Pass Manager Visualizations
===========================

.. autosummary::
   :toctree: ../stubs/

   pass_manager_drawer

Pulse Visualizations
====================

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.visualization.pulse_v2.draw ~qiskit.visualization.pulse_v2.IQXStandard
   ~qiskit.visualization.pulse_v2.IQXSimple ~qiskit.visualization.pulse_v2.IQXDebugging

Timeline Visualizations
=======================

.. autosummary::
   :toctree: ../stubs/

   timeline_drawer ~qiskit.visualization.timeline.draw

Single Qubit State Transition Visualizations
============================================

.. autosummary::
   :toctree: ../stubs/

   visualize_transition

Array/Matrix Visualizations
===========================

.. autosummary::
   :toctree: ../stubs/

   array_to_latex

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   VisualizationError
"""

import os
import sys
import warnings

from qiskit.visualization.counts_visualization import plot_histogram
from qiskit.visualization.state_visualization import (
    plot_state_hinton,
    plot_bloch_vector,
    plot_bloch_multivector,
    plot_state_city,
    plot_state_paulivec,
    plot_state_qsphere,
)
from qiskit.visualization.transition_visualization import visualize_transition
from qiskit.visualization.array import array_to_latex

from .circuit_visualization import circuit_drawer
from .dag_visualization import dag_drawer
from .exceptions import VisualizationError
from .gate_map import plot_gate_map, plot_circuit_layout, plot_error_map, plot_coupling_map
from .pass_manager_visualization import pass_manager_drawer
from .pulse.interpolation import step_wise, linear, cubic_spline
from .pulse.qcstyle import PulseStyle, SchedStyle
from .pulse_visualization import pulse_drawer
from .pulse_v2 import draw as pulse_drawer_v2
from .timeline import draw as timeline_drawer

_DEPRECATED_NAMES = {
    "HAS_MATPLOTLIB",
    "HAS_PYLATEX",
    "HAS_PIL",
    "HAS_PDFTOCAIRO",
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        from qiskit.utils import optionals

        warnings.warn(
            f"Accessing '{name}' from '{__name__}' is deprecated since Qiskit Terra 0.21 "
            "and will be removed in a future release. Use 'qiskit.utils.optionals' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(optionals, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
