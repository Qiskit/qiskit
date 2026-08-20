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

Graph-based circuit representations
===================================

While :class:`.QuantumCircuit` is the user-facing circuit model, much of Qiskit's
compiler stack works on graph-based intermediate representations defined in this
module.  A transpiler :class:`.PassManager` converts an input circuit into a
:class:`.DAGCircuit`, runs a sequence of analysis and transformation passes on that
graph, and converts the result back into a :class:`.QuantumCircuit` before returning
it.  See :mod:`qiskit.transpiler` for the broader compilation model.

The two primary graph types are :class:`DAGCircuit` and :class:`DAGDependency`.
Both store the operations from a circuit and the relationships between them, but
they emphasize different structure:

:class:`DAGCircuit`
    Models the circuit as wires flowing between operations.  Nodes are input nodes,
    output nodes, or operation nodes; edges follow qubits and classical bits from
    the output of one operation to the input of the next.  This is the default IR
    used throughout the transpiler because most passes need to preserve wire order
    and reason about data flow along individual qubits or clbits.

:class:`DAGDependency`
    Models dependencies induced by non-commutativity.  Edges indicate when one
    operation cannot be commuted past another, even if it acts on different qubits.
    This representation is used by passes that care about partial ordering rather
    than exact wire topology, such as commutation-aware block collection and some
    optimization routines.

Node types
==========

:class:`DAGCircuit` exposes :class:`DAGOpNode`, :class:`DAGInNode`, and
:class:`DAGOutNode`.  Operation nodes carry the underlying :class:`.Instruction`
(or :class:`.Gate`) and their qargs/cargs; input and output nodes mark where each
bit enters or leaves the graph.

:class:`DAGDependency` uses :class:`DAGDepNode` for its operation nodes.

Typical workflow
================

Conversion helpers live in :mod:`qiskit.converters`.  A common pattern is to build
or receive a :class:`.QuantumCircuit`, convert it to a DAG, inspect or transform
the graph, and convert back:

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag, dag_to_circuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        print(node.op.name, node.qargs)

    roundtrip = dag_to_circuit(dag)

When a pass needs commutation structure, use :func:`.circuit_to_dagdependency` and
:class:`DAGDependency` instead.  The converter functions in :mod:`qiskit.converters`
also provide conversions between the two DAG forms.

Block utilities
===============

Several transpiler passes split a DAG into contiguous blocks of operations, rewrite
those blocks, and stitch the result back together.  The :class:`BlockCollector`,
:class:`BlockSplitter`, and :class:`BlockCollapser` classes implement that pattern
for both :class:`DAGCircuit` and :class:`DAGDependency`.  Collecting blocks from
:class:`DAGDependency` can yield more optimization opportunities because it respects
commutation, at the cost of building the dependency graph first.

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
