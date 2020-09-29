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
=====================================
ClassicalFunction compiler (:mod:`qiskit.circuit.classical_function`)
=====================================

.. currentmodule:: qiskit.circuit.classical_function

Overview
========

The ClassicalFunction compiler provides the necessary tools to map a classical
irreversible functions into quantum circuits.  Below is a simple example of
how to synthesize a simple boolean function defined using Python into a
QuantumCircuit:

   .. jupyter-execute::

      from qiskit.circuit.classical_function import classical_function
      from qiskit.circuit.classical_function.types import Int1

      @classical_function
      def grover_oracle(a: Int1, b: Int1, c: Int1, d: Int1) -> Int1:
          return (not a and b and not c and d)

      quantum_circuit = grover_oracle.synth()
      quantum_circuit.draw()

Following Qiskit's little-endian bit ordering convention, the left-most bit (`a`) is the most
significant bit and the right-most bit (`d`) is the least significant bit.

Supplementary Information
=========================

.. container:: toggle

   .. container:: header

      **Tweedledum**

   Tweedledum is a C++-17 header-only library that implements a large set of
   reversible (and quantum) synthesis, optimization, and mapping algorithms.
   The classical_function compiler relies on it and its dependencies to both represent logic
   networks and synthesize them into quantum circuits.

.. container:: toggle

   .. container:: header

      **ClassicalFunction data types**

   At the moment, the only type supported by the classical_function compilers is
   ``qiskit.circuit.classical_function.types.Int1``. The classical_function function
   to parse *must* include type hints (just ``Int1`` for now).

   The type ``Int1`` means the classical_function will only operate at bit level.


ClassicalFunction compiler API
===================

classical_function
------

Alias for ``qiskit.circuit.classical_function.compile_classical_function.compile_classical_function``.
It can be used as a decorator.


ClassicalFunction
------

.. autosummary::
   :toctree: ../stubs/

   ClassicalFunction

Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   ClassicalFunctionCompilerTypeError
   ClassicalFunctionParseError
   ClassicalFunctionCompilerTypeError

"""
from .exceptions import (ClassicalFunctionParseError, ClassicalFunctionCompilerError,
                         ClassicalFunctionCompilerTypeError)
from .compile_classical_function import compile_classical_function as classical_function
from .types import Int1, Int2
