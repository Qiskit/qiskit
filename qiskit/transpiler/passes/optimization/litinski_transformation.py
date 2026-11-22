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
    r"""Applies the Litinski transform to a circuit.

    The transform applies to circuits containing Clifford gates, single-qubit
    :math:`R_X`, :math:`R_Y` and :math:`R_Z` rotations
    (including :math:`Phase`, :math:`T` and :math:`T^\dagger`), Pauli product rotations,
    Pauli product measurements, and standard :math:`Z`-measurements.

    It moves Clifford gates to the end of the circuit, converting
    single-qubit rotations to Pauli product rotations, and measurements to Pauli product
    measurements.

    Non-Clifford Pauli rotations in the output circuit are represented either as
    :class:`.PauliProductRotationGate` or as :class:`.PauliEvolutionGate` objects,
    depending on ``use_ppr``.

    If ``combine_paulis=True``, the pass additionally merges Pauli rotations
    with the same Pauli when all intervening rotations and measurements commute with that Pauli.
    Merged identity rotations are removed, and merged Clifford rotations are absorbed
    into the final Clifford.

    The pass supports all of the Clifford gates in the list returned by
    :func:`.get_clifford_gate_names`:

    ``["id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy",
    "swap","iswap", "ecr", "dcx"]``

    In addition, the rotation gates with angles that are integral multiples of :math:`\pi/2`
    (within the given tolerance) are considered Clifford.


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

    The following example illustrates the ``combine_paulis`` option. After moving the Clifford
    gates to the end of the circuit, the two :math:`T`-gates become rotations with the same
    Pauli, and are merged into a single :math:`\pi/2`-rotation. This rotation is a Clifford,
    and is hence added to the final Clifford gates, so that the output circuit contains no
    non-Clifford rotations.

    .. plot::
        :include-source:
        :nofigs:

        from qiskit.circuit import QuantumCircuit
        from qiskit.transpiler.passes import LitinskiTransformation

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.t(1)
        qc.h(0)
        qc.t(1)

        litinski = LitinskiTransformation(use_ppr=True, combine_paulis=True, insert_barrier=True)
        qct = litinski(qc)

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
            fix_clifford: If ``False``, omit the final Clifford gates from the
                output circuit. The resulting circuit is generally not operator-equivalent
                to the input circuit.
            insert_barrier: If ``True`` and ``fix_clifford=True``, insert a barrier before the
                final Clifford gates. Has no effect otherwise.
            use_ppr: Controls the representation of non-Clifford Pauli rotations in the output
                circuit. If ``True``, use :class:`.PauliProductRotationGate` objects; otherwise,
                use :class:`.PauliEvolutionGate` objects. This option does not affect Clifford gates
                or Pauli product measurements. Using ``True`` is recommended.
            approximation_degree: Tolerance used when determining whether a rotation is sufficiently
                close to a Clifford, based on average gate fidelity.
            combine_paulis: combine_paulis: If ``True``, combine compatible Pauli product rotations,
                exploiting commutativity between them.
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
