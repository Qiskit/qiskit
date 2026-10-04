# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=======================================
DAG Circuits (:mod:`qiskit.dagcircuit`)
=======================================

.. currentmodule:: qiskit.dagcircuit

The Directed Acyclic Graph (DAG) representation of a quantum circuit is a core data structure used heavily by the transpiler. 

Unlike the :class:`~qiskit.circuit.QuantumCircuit` representation, which is optimized for user construction and viewing, the DAG representation models operations as nodes in a graph with directed edges representing the flow of qubits and classical bits between them. This allows the transpiler to easily query dependencies, identify commutative operations, and safely reorder or optimize gates by leveraging graph theory algorithms.

The module provides the main :class:`DAGCircuit` object, as well as various node types (:class:`DAGOpNode`, :class:`DAGInNode`, :class:`DAGOutNode`) that represent the components of the graph.

Circuits as Directed Acyclic Graphs
===================================

.. autosummary::
   :toctree: ../stubs/

   DAGCircuit
   DAGNode
   DAGOpNode
   DAGInNode
   DAGOutNode
   DAGDepNode
   DAGDependency

Exceptions
==========

.. autoexception:: DAGCircuitError
.. autoexception:: DAGDependencyError

Utilities
=========

.. autosummary::
   :toctree: ../stubs/

   BlockCollapser
   BlockCollector
   BlockSplitter
"""
from .collect_blocks import BlockCollapser, BlockCollector, BlockSplitter
from .dagcircuit import DAGCircuit
from .dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from .dagdepnode import DAGDepNode
from .exceptions import DAGCircuitError, DAGDependencyError
from .dagdependency import DAGDependency

__all__ = [
    "BlockCollapser",
    "BlockCollector",
    "BlockSplitter",
    "DAGCircuit",
    "DAGCircuitError",
    "DAGDepNode",
    "DAGDependency",
    "DAGDependencyError",
    "DAGInNode",
    "DAGNode",
    "DAGOpNode",
    "DAGOutNode",
]
