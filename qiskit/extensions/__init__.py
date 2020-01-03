# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================================
Quantum Circuit Extensions (:mod:`qiskit.extensions`)
=====================================================

.. currentmodule:: qiskit.extensions

Standard Extensions
===================

.. autosummary::
   :toctree: ../stubs/

   Barrier
   ToffoliGate
   CHGate
   CrxGate
   CryGate
   CrzGate
   FredkinGate
   Cu1Gate
   Cu3Gate
   CnotGate
   CyGate
   CzGate
   HGate
   IdGate
   MSGate
   RXGate
   RXXGate
   RYGate
   RZGate
   RZZGate
   SGate
   SdgGate
   SwapGate
   TdgGate
   U0Gate
   U1Gate
   U2Gate
   U3Gate
   XGate
   YGate
   ZGate

Unitary Extensions
==================

.. autosummary::
   :toctree: ../stubs/

   UnitaryGate

Simulator Extensions
====================

.. autosummary::
   :toctree: ../stubs/

   Snapshot

Initialization
==============

.. autosummary::
   :toctree: ../stubs/

   Initialize
"""

from qiskit.extensions.quantum_initializer.initializer import Initialize
from .standard import *
from .unitary import UnitaryGate
from .simulator import Snapshot
