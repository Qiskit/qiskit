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

"""Initialize the Jupyter routines.
"""

from IPython import get_ipython          # pylint: disable=import-error
from qiskit.tools.visualization import HAS_MATPLOTLIB
from .jupyter_magics import (ProgressBarMagic, StatusMagic)
from .progressbar import HTMLProgressBar

if HAS_MATPLOTLIB:
    from .backend_overview import BackendOverview
    from .backend_monitor import BackendMonitor

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(ProgressBarMagic)
    _IP.register_magics(StatusMagic)
    if HAS_MATPLOTLIB:
        _IP.register_magics(BackendOverview)
        _IP.register_magics(BackendMonitor)
