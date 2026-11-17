# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Move clifford gates to the end of the circuit, changing rotation gates to multi-qubit rotations."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.litinski_transformation import run_litinski_transformation


class LitinskiTransformation(TransformationPass):
    r"""Applies Litinski transform to a circuit.

    The transform applies to a circuit containing Clifford, single-qubit :math:`R_Z`-rotation,
    :math:`R_X`-rotation and :math:`R_Y`-rotation gates,
    (including :math:`Phase`, :math:`T` and :math:`T^\dagger`), Pauli product rotations,
    Pauli product measurements, and standard :math:`Z`-measurements.
    The transform moves Clifford gates to the end of the circuit, including single-qubit rotation gates,
    Pauli product rotations and Pauli product measurements, whose angle is a multiple of :math:`\pi/2`.
    In the process, it transforms :math:`R_Z`-rotations,
    :math:`R_X`-rotation and :math:`R_Y`-rotation gates to
    Pauli product rotations, and :math:`Z`-measurements to Pauli product measurements.
    The Pauli product rotations in the output circuit are represented either as
    :class:`.PauliProductRotationGate` or as :class:`.PauliEvolutionGate` objects,
    depending on the ``use_ppr`` argument; this applies uniformly to all of the
    non-Clifford rotations, including the Pauli product rotations of the input circuit.

    Optionally, when ``combine_paulis=True``, the pass additionally reduces the number of
    non-Clifford rotations by merging rotations with the same Pauli (after moving the Clifford
    gates to the end of the circuit) that are separated only by commuting rotations and
    measurements. Merged rotations that are equivalent to the identity are removed, and merged
    rotations that are Clifford are absorbed into the final Clifford.

    The pass supports all of the Clifford gates in the list returned by
    :func:`.get_clifford_gate_names`:

    ``["id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy",
    "swap","iswap", "ecr", "dcx"]``

    In addition, the rotation gates above with angles that are integral multiples of :math:`\pi/2`
    within the given tolerance are also considered Clifford.


    Example:

    .. plot::
        :include-source:
        :nofigs:

        from qiskit import generate_preset_pass_manager
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PauliProductMeasurement, PauliProductRotationGate
        from qiskit.quantum_info import Pauli
        from qiskit.transpiler.passes import LitinskiTransformation

        litinski = LitinskiTransformation(fix_clifford=False, use_ppr=True)

        rz_basis = ["rz", "h", "x", "cx"]
        pm = generate_preset_pass_manager(basis_gates=rz_basis)
        pm.optimization.append(litinski)

        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.rz(1.23, 0)
        qc.cx(0, 1)
        qc.t(1)
        qc.append(PauliProductRotationGate(Pauli("XY"), 0.456), [1, 2])
        qc.cx(1, 2)
        qc.append(PauliProductMeasurement(Pauli("ZX")), [0, 1], [0])
        qc.measure(2, 0)


        pbc = pm.run(qc)

    References:

    [1] Litinski. A Game of Surface Codes.
    `Quantum 3, 128 (2019) <https://quantum-journal.org/papers/q-2019-03-05-128>`_

    """

    def __init__(
        self,
        fix_clifford: bool = True,
        insert_barrier: bool = False,
        use_ppr: bool | None = None,
        approximation_degree: float = 1.0,
        combine_paulis: bool = False,
    ):
        r"""
        Args:
            fix_clifford: If ``False`` (non-default), the returned circuit contains
                only the Pauli rotations and Pauli product measurements, with the final Clifford
                gates omitted. Note that in this case the operators of the original and
                synthesized circuits will generally not be equivalent.
            insert_barrier: If ``True`` and ``fix_clifford=True``, insert a barrier between the
                circuit and the final cliffords. This argument has no effect if
                ``fix_clifford=False``.
            use_ppr: Controls how the non-Clifford Pauli rotations are represented in the output
                circuit. This applies to all such rotations: those obtained from the single-qubit
                rotation gates (including :math:`T` and :math:`T^\dagger`), the non-Clifford
                :class:`.PauliProductRotationGate` objects of the input circuit, and (with
                ``combine_paulis=True``) the merged rotations. If ``True``, the rotations are
                represented as :class:`.PauliProductRotationGate` objects. This is encouraged
                to improve performance using a fully Rust-backed path. If ``False`` or ``None``,
                the rotations are represented as :class:`.PauliEvolutionGate` objects, including
                those that were :class:`.PauliProductRotationGate` objects in the input circuit.
                This option does not affect the final Clifford gates, which are the Clifford
                gates of the original circuit (possibly including
                :class:`.PauliProductRotationGate` objects with angles that are multiples of
                :math:`\pi/2`), followed, when ``combine_paulis=True``, by the merged rotations
                that are Clifford, which are always represented as
                :class:`.PauliProductRotationGate` objects. This option also does not affect the
                Pauli product measurements.
            approximation_degree: Used in the tolerance computations,
                to check how much a PPR or a rotation gate is close to a Clifford.
                This gives the threshold for the average gate fidelity.
            combine_paulis: If ``True``, additionally combines the non-Clifford Pauli rotations
                produced by the transform, exploiting commutativity between Pauli product
                rotations and Pauli product measurements. Specifically, each new rotation
                (after being moved past the Clifford gates) is merged with an earlier rotation
                with the same Pauli, provided that it commutes with all of the rotations and
                measurements in between, by summing the two angles. This applies to all of the
                rotations, whether they originate from single-qubit rotation gates or from
                :class:`.PauliProductRotationGate` objects in the input circuit. If the merged
                rotation is equivalent to the identity (up to a global phase), it is removed
                and the global phase is updated accordingly. If the merged rotation is a
                Clifford (that is, its angle is a multiple of :math:`\pi/2`), it is removed
                and absorbed into the final Clifford; when ``fix_clifford=True``, it is added
                to the final Clifford gates as a :class:`.PauliProductRotationGate` (regardless
                of the value of ``use_ppr``).
                Note that this option may increase the running time of the pass: when an
                earlier rotation with the same Pauli exists, each new rotation is compared
                against the previously collected rotations and measurements that commute with
                it. In the worst case, where rotations with the same Pauli repeatedly appear far
                apart in the circuit, separated by many commuting rotations, the running time is
                quadratic in the number of non-Clifford rotations.
        """
        super().__init__()
        self.fix_clifford = fix_clifford
        self.insert_barrier = insert_barrier
        self.approximation_degree = approximation_degree
        self.combine_paulis = combine_paulis

        # In Qiskit v2.4 the default is to keep using PauliEvolutionGate as rotation gates, but
        # come v2.5 we can start to warn that in v3.0 the default will be changed to PPR gates
        # (i.e. we will set ``use_ppr=True`` per default).
        if use_ppr is None:
            use_ppr = False
        self.use_ppr = use_ppr

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the LitinskiTransformation pass on ``dag``.

        Args:
            dag: The input DAG.

        Returns:
            The output DAG.

        Raises:
            TranspilerError: If the circuit contains gates not supported by the pass.
        """
        new_dag = run_litinski_transformation(
            dag,
            self.fix_clifford,
            self.insert_barrier,
            self.use_ppr,
            self.combine_paulis,
            self.approximation_degree,
        )

        # If the pass did not do anything, the result is None
        if new_dag is None:
            return dag

        return new_dag
