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
Oracle compiler (:mod:`qiskit.transpiler.oracle_compiler`)
=====================================

.. currentmodule:: qiskit.transpiler.oracle_compiler

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

      from qiskit.transpiler.oracle_compiler import compile_oracle
      from qiskit.transpiler.oracle_compiler.types import Bit

      def grover_oracle(a: Bit, b: Bit, c: Bit, d: Bit) -> Bit:
          return (not a and b and not c and d)

      network = compile_oracle(grover_oracle)
      quantum_circuit = network.synth()
      print(quantum_circuit)

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
   ``qiskit.transpiler.oracle_compiler.types.Bit``. The oracle function
   to parse *must* include type hints (just ``Bit`` for now).

   The type ``Bit`` means the oracle will only operate at bit level.


Oracle compiler API
===================

Logic Network
-------------

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
from .compile_oracle import compile_oracle
