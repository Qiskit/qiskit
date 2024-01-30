# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===============================================================================
BasicProvider: Python-based Simulators (:mod:`qiskit.providers.basic_provider`)
===============================================================================

.. currentmodule:: qiskit.providers.basic_provider

A module of Python-based quantum simulators. Simulators can be accessed
via the `BasicProvider` provider, e.g.:

.. code-block::

   from qiskit.providers.basic_provider import BasicProvider

   backend = BasicProvider().get_backend('basic_simulator')


Simulators
==========

.. autosummary::
   :toctree: ../stubs/

   BasicSimulator

Provider
========

.. autosummary::
   :toctree: ../stubs/

   BasicProvider

Job Class
=========

.. autosummary::
   :toctree: ../stubs/

   BasicProviderJob

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   BasicProviderError
"""

from .basic_provider import BasicProvider
from .basic_provider_job import BasicProviderJob
from .basic_simulator import BasicSimulator
from .exceptions import BasicProviderError
