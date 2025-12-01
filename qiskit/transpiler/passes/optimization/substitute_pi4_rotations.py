# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Convert rotation gates into {Clifford,T,Tdg} when their angles are integer multiples of pi/4"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.substitute_pi4_rotations import substitute_pi4_rotations


class SubstitutePi4Rotations(TransformationPass):
    r"""Convert single-qubit rotation gates :class:`.RZGate`, :class:`.RXGate` and :class:`.RYGate`,
    whose angles are integer multiples of :math:`\pi/4` into discrete sets of
    Clifford, :class:`.TGate` and :class:`.TdgGate` gates.

    Note that odd multiples of :math:`\pi/4` require a single :class:`.TGate` or :class:`.TdgGate`,
    as well as some Clifford gates,
    while even multiples of :math:`\pi/4`, or equivalently, integer multiples of :math:`\pi/2`,
    can be written using only Clifford gates.
    The output contains at most one :class:`.TGate` or :class:`.TdgGate`,
    and an optimal number of Clifford gates.

    For example::

      from qiskit.circuit import QuantumCircuit
      from qiskit.transpiler.passes import SubstitutePi4Rotations
      from qiskit.quantum_info import Operator

      # The following quantum circuit consists of 5 Clifford gates
      # and three single-qubit rotation gates whose angles are integer multiples of pi/4.

      qc = QuantumCircuit(3)
      qc.cx(0, 1)
      qc.rz(pi/4, 0)
      qc.cz(0, 1)
      qc.rx(3*pi/4, 1)
      qc.h(1)
      qc.s(2)
      qc.ry(2*pi/4, 2)
      qc.cz(2, 0)

      # The transformed circuit consists of Clifford, T and Tdg gates
      qct = SubstitutePi4Rotations()(qc)
      clifford_t_names = get_clifford_gate_names() + ["t"] + ["tdg"]
      assert(set(qct.count_ops().keys()).issubset(set(clifford_t_names)))

      # The circuits before and after the transformation are equivalent
      assert Operator(qc) == Operator(qct)
    """

    def __init__(self, approximation_degree: float = 1.0):
        """
        Args:
            approximation_degree: Used in the tolerance computations.
                This gives the threshold for the average gate fidelity.
        """
        super().__init__()
        self.approximation_degree = approximation_degree

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Substitute Pi4-Rotations optimization pass on ``dag``.

        Args:
            dag: the input DAG.

        Returns:
            The output DAG.
        """
        new_dag = substitute_pi4_rotations(dag, self.approximation_degree)

        # If the pass did not do anything, the result is None
        if new_dag is None:
            return dag

        return new_dag
