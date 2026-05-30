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

This module provides the directed acyclic graph (DAG) representation of quantum circuits.
A :class:`~qiskit.dagcircuit.DAGCircuit` represents the operations and dependencies within
a quantum circuit as a graph. This format is primarily used by the Qiskit transpiler
to analyze, route, and optimize quantum circuits.

In a DAG circuit, nodes represent operations (gates, measurements, resets) or input/output
wires, while directed edges represent the flow of quantum or classical data between these nodes.
The edges ensure that dependencies are strictly preserved during any transformations.

Using the DAG representation
============================

You can convert between a :class:`~qiskit.circuit.QuantumCircuit` and a
:class:`~qiskit.dagcircuit.DAGCircuit` using the converter functions provided in
:mod:`qiskit.converters`.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag, dag_to_circuit

    # Create a circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Convert to DAG
    dag = circuit_to_dag(qc)

    # Explore the nodes
    for node in dag.op_nodes():
        print(node.name)

    # Convert back to QuantumCircuit
    optimized_qc = dag_to_circuit(dag)

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
