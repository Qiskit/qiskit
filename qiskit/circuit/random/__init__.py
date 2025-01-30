# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
==============================================
Random Circuits (:mod:`qiskit.circuit.random`)
==============================================

.. currentmodule:: qiskit.circuit.random

Overview
========

Existing architecture of Quantum Computers have varying computational capabilities.
High number of highly connected qubits with lower gate error rates, and faster gate 
times are defining properties of a capable Quantum Computer.

One of the basic usages of a quantum circuit with arbitrary gates, qubits, depth etc
is to benchmark existing Quantum Hardware. It can also be used to estimate the 
performance of quantum circuit transpilers and supporting software infrastructure.

Below functions can be used to generate an arbitrary circuit with gates randomly 
selected from a given set of gates. These functions can generate bespoke quantum 
circuits respecting properties like number of qubits, depth of circuit, coupling map
of the hardware, gate set, etc.

Generating arbitrary circuits respecting qubit-coupling
--------------------------------------------------------------

.. autofunction:: random_circuit_from_graph


Generating arbitrary circuits
------------------------------------

.. autofunction:: random_circuit


Generating arbitrary circuits with clifford gates
--------------------------------------------------------

.. autofunction:: random_clifford_circuit

"""

from .utils import random_circuit, random_clifford_circuit, random_circuit_from_graph
