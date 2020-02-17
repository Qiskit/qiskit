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
========================================
Quantum Circuits (:mod:`qiskit.circuit`)
========================================

.. currentmodule:: qiskit.circuit

Quantum Circuit Construction
============================

.. autosummary::
   :toctree: ../stubs/

   QuantumCircuit
   QuantumRegister
   Qubit
   ClassicalRegister
   Clbit

Gates and Instructions
======================

.. autosummary::
   :toctree: ../stubs/

   Gate
   ControlledGate
   Measure
   Reset
   Instruction
   InstructionSet

Parametric Quantum Circuits
===========================

.. autosummary::
   :toctree: ../stubs/

    Parameter
    ParameterVector
    ParameterExpression

"""
from .quantumcircuit import QuantumCircuit
from .classicalregister import ClassicalRegister, Clbit
from .quantumregister import QuantumRegister, Qubit
from .gate import Gate
from .controlledgate import ControlledGate
from .instruction import Instruction
from .instructionset import InstructionSet
from .measure import Measure
from .reset import Reset
from .parameter import Parameter
from .parametervector import ParameterVector
from .parameterexpression import ParameterExpression
