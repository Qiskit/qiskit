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
===========================================
Jupyter Tools (:mod:`qiskit.tools.jupyter`)
===========================================

.. currentmodule:: qiskit.tools.jupyter

A Collection of Jupyter magic functions and tools
that extend the functionality of Qiskit.

Overview of all available backends
==================================

.. code-block::

    from qiskit.providers.ibmq import IBMQ
    import qiskit.tools.jupyter
    %matplotlib inline

    IBMQ.load_account()

    %qiskit_backend_overview


Detailed information on a single backend
========================================

.. code-block::

    from qiskit.providers.ibmq import IBMQ
    import qiskit.tools.jupyter
    %matplotlib inline

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    backend = provider.get_backend('ibmq_vigo')
    backend


Load Qiskit Job Watcher
=======================

.. code-block::

    import qiskit.tools.jupyter
    %qiskit_job_watcher


HTMLProgressBar
===============

.. code-block::

    import numpy as np
    from qiskit.tools.parallel import parallel_map
    import qiskit.tools.jupyter

    %qiskit_progress_bar
    parallel_map(np.sin, np.linspace(0,10,100));


Qiskit version table
====================

.. code-block::

    import qiskit.tools.jupyter
    %qiskit_version_table


Qiskit copyright
================

.. code-block::

    import qiskit.tools.jupyter
    %qiskit_copyright

Monospaced output
=================

.. code-block::

    import qiskit.tools.jupyter
    %monospaced_output

"""
import warnings

from IPython import get_ipython
from qiskit.providers.fake_provider import FakeBackend
from qiskit.utils import optionals as _optionals
from .jupyter_magics import ProgressBarMagic, StatusMagic
from .progressbar import HTMLProgressBar
from .version_table import VersionTable
from .copyright import Copyright
from .monospace import MonospacedOutput
from .job_watcher import JobWatcher, JobWatcherMagic

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(ProgressBarMagic)
    _IP.register_magics(VersionTable)
    _IP.register_magics(MonospacedOutput)
    _IP.register_magics(Copyright)
    _IP.register_magics(JobWatcherMagic)
    if _optionals.HAS_MATPLOTLIB:
        from .backend_overview import BackendOverview
        from .backend_monitor import _backend_monitor

        _IP.register_magics(BackendOverview)
        if _optionals.HAS_IBMQ:
            from qiskit.providers.ibmq import IBMQBackend  # pylint: disable=no-name-in-module

            HTML_FORMATTER = _IP.display_formatter.formatters["text/html"]
            # Make _backend_monitor the html repr for IBM Q backends
            HTML_FORMATTER.for_type(IBMQBackend, _backend_monitor)
            HTML_FORMATTER.for_type(FakeBackend, _backend_monitor)
    else:
        warnings.warn(
            "matplotlib can't be found, ensure you have matplotlib and other "
            "visualization dependencies installed. You can run "
            "'!pip install qiskit-terra[visualization]' to install it from "
            "jupyter",
            RuntimeWarning,
        )
