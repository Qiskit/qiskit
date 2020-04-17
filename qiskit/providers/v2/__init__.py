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
provider is anything that provides an external service to Terra.


Abstract Classes
================

.. autosummary::
   :toctree: ../stubs/

   Provider
   Backend
   Configuration
   Properties
"""

from .provider import Provider
from .backend import Backend
from .configuration import Configuration
from .properties import Properties
from .job import Job
