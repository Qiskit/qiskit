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
=================

.. autosummary::
   :toctree: ../stubs/

   parallel_map - Parallel execution of a mapping of `values` to the function `task`.

Monitoring
==========

.. autosummary::
   :toctree: ../stubs/

   job_monitor - Monitor the status of a IBMQJob instance
   backend_monitor - Monitor a single IBMQ backend
   backend_overview - Gives overview information on all the IBMQ backends that are available

"""

from .parallel import parallel_map
from .monitor import job_monitor, backend_monitor, backend_overview
