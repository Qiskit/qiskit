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

"""Synthesize RZ gates to Clifford+T efficiently"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.synthesize_rz_rotations import synthesize_rz_rotations


class SynthesizeRZRotations(TransformationPass):
    """Replace RZ gates with Clifford+T decompositions in an efficient manner.

    This pass replaces all single-qubit RZ rotation gates with sequences
    of Clifford+T gates. We first canonicalize angles based on the 4π cyclicity
    of RZ gates, and further utilize properties of RZ(θ) to limit the number of
    distinct angles synthesized to [0, π/2) and a u8 bit.

    This pass performs a two-fold optimization for implementing RZ synthesis.
    For QFT-like circuits with repeating angles, we want to avoid doing redundant
    syntheses of the same angle multiple times. The mechanism for this is to first
    collect all the angles of RZ gates from the DAG and sort them. If we implement
    a cache-like mechanism where if the synthesis angle already exists, we reuse it,
    if it does not, then we synthesize them. We further extend this by allowing
    approximations of angles and doing the synthesis with a lower epsilon.
    Because RZ is  4π-cyclic, we can come up with canonical representations of angles
    from  (−∞, ∞) → [0, 4π) . Further, we can exploit properties of  RZ  such as:
    RZ(θ+π/2) = RZ(θ)⋅S , RZ(θ+π) = RZ(θ)⋅Z, RZ(θ+2π) = −RZ(θ) to partition the [0, 4π)
    domain as ∪ [nπ/2, π(n+1)/2] , n ∈ Z + : [0, 7]. The canonical angle representation
    reduces to range [0 ,π/2) for RZ synthesis, but we make updates to the global phase
    and gates to add to the synthesised  RZ sequence.
    .
    Hence, this mapping further reduces as:
    [0, 4π) → {{r : r ∈ Z + , [0, 7]} , (0, π/2]}, where r corresponds to a specific
    phase and gate update (from a static look-up table) to be made to the DAG.

    We then iterate over the dag to identify RZ gates and replace them with their
    Clifford+T approximations.

    For example::

      from qiskit.circuit import QuantumCircuit
      from qiskit.transpiler.passes import SynthesizeRZRotations
      from qiskit.quantum_info import Operator
      from numpy import pi

      # The following quantum circuit consists of 5 Clifford gates
      # and three single-qubit RZ rotations gates

      qc = QuantumCircuit(4)
      qc.cx(0, 1)
      qc.rz(9*pi/4, 0)
      qc.cz(0, 1)
      qc.rz(3*pi/8, 1)
      qc.h(1)
      qc.s(2)
      qc.rz(13*pi/2, 2)
      qc.cz(2, 0)
      qc.rz(5*pi/3, 3)

      qct = SynthesizeRZRotations()(qc)

      # The transformed circuit consists of Clifford, T and Tdg gates
      clifford_t_names = get_clifford_gate_names() + ["t"] + ["tdg"]
      assert(set(qct.count_ops().keys()).issubset(set(clifford_t_names)))

      # The circuits before and after the transformation are equivalent
      assert Operator(qc) == Operator(qct)

    """

    def __init__(self, approximation_degree: float = 0.9999999999):
        """
        Args:t
        approximation_degree: float = 0.9999999999
        """
        super().__init__()
        self.approximation_degree = approximation_degree

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on a DAG.
        Args:
            dag: the input DAG.
        Returns:
            The output DAG.
        """
        new_dag = synthesize_rz_rotations(dag, self.approximation_degree)

        if new_dag is None:
            return dag

        return new_dag
