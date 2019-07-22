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
from .version_table import VersionTable
from .copyright import Copyright
from .job_watcher import JobWatcher, JobWatcherMagic

if HAS_MATPLOTLIB:
    from .backend_overview import BackendOverview
    from .backend_monitor import _backend_monitor

try:
    from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
    HAS_IBMQ = True
except ImportError:
    HAS_IBMQ = False


_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(ProgressBarMagic)
    _IP.register_magics(VersionTable)
    _IP.register_magics(Copyright)
    _IP.register_magics(JobWatcherMagic)
    if HAS_MATPLOTLIB:
        _IP.register_magics(BackendOverview)
        if HAS_IBMQ:
            HTML_FORMATTER = _IP.display_formatter.formatters['text/html']
            # Make _backend_monitor the html repr for IBM Q backends
            HTML_FORMATTER.for_type(IBMQBackend, _backend_monitor)
