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
Oracle compiler (:mod:`qiskit.circuit.oracle_compiler`)
=====================================

.. currentmodule:: qiskit.circuit.oracle_compiler

Overview
========
TODO

   .. jupyter-execute::

      from qiskit.circuit.oracle_compiler import compile_oracle
      from qiskit.circuit.oracle_compiler.types import Bit

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

   TODO, introduce to Tweedledum

.. container:: toggle

   .. container:: header

      **Oracle data types**

   TODO, introduce to Bit type ``qiskit.circuit.oracle_compiler.types``


Oracle compiler API
===================

Logic Network
-------------

.. autosummary::
   :toctree: ../stubs/

   LogicNetwork

Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   OracleCompilerTypeError
   OracleParseError
   OracleCompilerTypeError

"""
from .logic_network import LogicNetwork
from .exceptions import OracleParseError, OracleCompilerError, OracleCompilerTypeError
from .compile_oracle import compile_oracle
