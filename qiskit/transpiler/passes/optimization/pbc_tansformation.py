# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Convert standard gates into Pauli product rotation gates for PBC"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.pbc_transformation import pbc_transformation


class PBCTransformation(TransformationPass):
    r"""
    Convert a quanutm circuit containing single-qubit and two-qubit standard gates,
    barriers and measurements, into an equivalent list of Pauli product rotations,
    implemented as :class:`.PauliEvolutionGate` and a global phase,
    as well as :class:`.PauliProductMeasurement`.

    For example::

      from qiskit.circuit import QuantumCircuit
      from qiskit.transpiler.passes import PBCTransformation
      from qiskit.quantum_info import Operator

      qc = QuantumCircuit(3)
      qc.h(0)
      qc.cx(0, 1)
      qc.ry(0.123, 0)
      qc.t(2)
      qc.rzz(pi/4, 0, 2)

      # The transformed circuit consists of PauliEvolution gates
      qct = PBCTransformation()(qc)
      ops_names = set(qct.count_ops().keys())
      self.assertEqual(ops_names, {"PauliEvolution"})

      # The circuits before and after the transformation are equivalent
      assert Operator(qc) == Operator(qct)
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the PBC Transformayion optimization pass on ``dag``.

        Args:
            dag: the input DAG.

        Returns:
            The output DAG.
        """
        dag = pbc_transformation(dag)

        return dag
