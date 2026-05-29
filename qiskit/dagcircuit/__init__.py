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

Qiskit's graph-based circuit representations
============================================

Qiskit uses directed acyclic graphs (DAGs) as intermediate representations of circuits for
transformations, analyses and optimization.  The two main public DAG types in this module are
:class:`DAGCircuit` and :class:`DAGDependency`.  Both represent the operations of a circuit and their
relationships, but they model different kinds of structure and are useful in different contexts.

Use :class:`DAGCircuit` when you need a circuit-like graph whose edges are the quantum and classical
wires flowing between operations.  This is the primary graph representation used throughout the
transpiler stack, and is the right choice for most analyses and transformations that care about wire
order, data flow, or rewrites that should preserve the step-by-step structure of the circuit.

Use :class:`DAGDependency` when you need to reason about dependencies induced by
non-commutativity.  Its edges represent when operations cannot commute past each other, which makes
it useful for passes and algorithms that care more about partial ordering than about the exact wire
graph.

The node classes exposed by this module are the typed views into these graph representations:

* :class:`DAGOpNode`, :class:`DAGInNode`, and :class:`DAGOutNode` are the node types used by
  :class:`DAGCircuit`.
* :class:`DAGDepNode` is the node type used by :class:`DAGDependency`.

In typical workflows, a :class:`.QuantumCircuit` is converted into one of these DAG forms, processed,
and then converted back into a circuit.  The :mod:`qiskit.converters` module provides the helper
functions for moving between these representations.

Circuits as directed acyclic graphs
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
