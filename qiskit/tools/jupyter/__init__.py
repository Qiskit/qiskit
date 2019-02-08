# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Initialize the Jupyter routines.
"""

from IPython import get_ipython          # pylint: disable=import-error
from qiskit.tools.visualization._matplotlib import HAS_MATPLOTLIB
from .jupyter_magics import (ProgressBarMagic, StatusMagic)
from .progressbar import HTMLProgressBar

if HAS_MATPLOTLIB:
    from ._backend_overview import BackendOverview
    from ._backend_monitor import BackendMonitor

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(ProgressBarMagic)
    _IP.register_magics(StatusMagic)
    if HAS_MATPLOTLIB:
        _IP.register_magics(BackendOverview)
        _IP.register_magics(BackendMonitor)
