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
=====================================
Oracle compiler (:mod:`qiskit.circuit.oracle`)
=====================================

.. currentmodule:: qiskit.circuit.oracle

Overview
========

Oracles play an essential role in different quantum algorithms, e.g., Grover's
algorithm.  An oracle is a "black box" with a reliable behavior.  Theoretically,
its implementation does not need to be specified.  However, to execute an
algorithm that relies on an oracle in the real world, we need a concrete
implementation.

The oracle compiler provides the necessary tools to map a classical
irreversible functions into quantum circuits.  Below is a simple example of a
how to synthesize a simple boolean function defined using Python into a
QuantumCircuit:

   .. jupyter-execute::

      from qiskit.circuit.oracle import oracle
      from qiskit.circuit.oracle.types import Int1

      @oracle
      def grover_oracle(a: Int1, b: Int1, c: Int1, d: Int1) -> Int1:
          return (not a and b and not c and d)

      quantum_circuit = grover_oracle.synth()
      quantum_circuit.draw()

Supplementary Information
=========================

.. container:: toggle

   .. container:: header

      **Tweedledum**

   Tweedledum is a C++-17 header-only library that implements a large set of
   reversible (and quantum) synthesis, optimization, and mapping algorithms.
   The oracle compiler relies on it and its dependencies to both represent logic
   networks and synthesize them into quantum circuits.

.. container:: toggle

   .. container:: header

      **Oracle data types**

   At the moment, the only type supported by the oracle compilers is
   ``qiskit.circuit.oracle.types.Int1``. The oracle function
   to parse *must* include type hints (just ``Int1`` for now).

   The type ``Int1`` means the oracle will only operate at bit level.


Oracle compiler API
===================

oracle
------

Alias for ``qiskit.circuit.oracle.compile_oracle.compile_oracle``.
It can be used as a decorator.


Oracle
------

.. autosummary::
   :toctree: ../stubs/

   Oracle

Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   OracleCompilerTypeError
   OracleParseError
   OracleCompilerTypeError

"""
from .oracle import Oracle
from .exceptions import OracleParseError, OracleCompilerError, OracleCompilerTypeError
from .compile_oracle import compile_oracle as oracle
