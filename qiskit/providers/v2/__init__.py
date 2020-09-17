# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
================================================
Providers Interface (:mod:`qiskit.providers.v2`)
================================================

.. currentmodule:: qiskit.providers.v2

This module contains the classes used to build external providers for Terra. A
provider is anything that provides an external service to Terra. The typical
example of this is a Backend provider which provides
:class:`~qiskit.providers.v2.Backend` objects which can be used for executing
:class:`~qiskit.circuits.QuantumCircuit` and/or :class:`~qiskit.pulse.Schedule`
objects. This contains the abstract classes which are used to define the
interface between a provider and terra.

Abstract Classes
================

Provider
--------

.. autosummary::
   :toctree: ../stubs/

   Provider
   ProviderV1

Backend
-------

.. autosummary::
   :toctree: ../stubs/

   Backend
   BackendV1

Options
-------

.. autosummary::
   :toctree: ../stubs/

   Options

Properties
----------

.. autosummary::
   :toctree: ../stubs/

   Properties
   PropertiesV1

Job
---

.. autosummary::
   :toctree: ../stubs/

   Job
   JobV1

ResultData
----------

.. autosummary::
   :toctree: ../stubs/

   ResultData
"""

from .provider import Provider
from .provider import ProviderV1
from .backend import Backend
from .backend import BackendV1
from .options import Options
from .properties import Properties
from .properties import PropertiesV1
from .job import Job
from .job import JobV1
from .result_data import ResultData
