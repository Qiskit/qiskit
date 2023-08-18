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
==================================
Qiskit Tools (:mod:`qiskit.tools`)
==================================

.. currentmodule:: qiskit.tools

Parallel Routines
-----------------

A helper function for calling a custom function with python ``ProcessPoolExecutor``.
Tasks can be executed in parallel using this function.
It has a built-in event publisher to show the progress of the parallel tasks.

.. autofunction:: parallel_map

Monitoring
----------

A helper module to get IBM backend information and submitted job status.

.. autofunction:: job_monitor
.. autofunction:: backend_monitor
.. autofunction:: backend_overview

.. automodule:: qiskit.tools.events

"""

from .parallel import parallel_map
from .monitor import job_monitor, backend_monitor, backend_overview
from .events import progressbar
