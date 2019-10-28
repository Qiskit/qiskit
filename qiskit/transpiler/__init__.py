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

"""
=====================================
Transpiler (:mod:`qiskit.transpiler`)
=====================================

.. currentmodule:: qiskit.transpiler

Circuit Transpilation
=====================

.. autosummary::
   :toctree: ../stubs/

   transpile_circuit

Pass Managment
==============

.. autosummary::
   :toctree: ../stubs/

   PassManager
   PropertySet
   FlowController

Layout and Topology
===================

.. autosummary::
   :toctree: ../stubs/

   Layout
   CouplingMap

Fenced Objects
==============

.. autosummary::
   :toctree: ../stubs/

   FencedDAGCircuit
   FencedPropertySet

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   TranspilerError
   TranspilerAccessError
"""

from .runningpassmanager import FlowController
from .passmanager import PassManager
from .propertyset import PropertySet
from .exceptions import TranspilerError, TranspilerAccessError
from .fencedobjs import FencedDAGCircuit, FencedPropertySet
from .basepasses import AnalysisPass, TransformationPass
from .coupling import CouplingMap
from .layout import Layout
from .transpile_circuit import transpile_circuit
