# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesize a single qubit gate to a discrete basis set."""

from __future__ import annotations

import typing
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import get_standard_gate_name_mapping, HGate, TGate, TdgGate
from qiskit.utils.deprecation import deprecate_func
from qiskit._accelerate.synthesis.discrete_basis import (
    SolovayKitaevSynthesis as RustSolovayKitaevCompiler,
    GateSequence,
)

from .generate_basis_approximations import _1q_gates

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit


class SolovayKitaevCompiler:
    """The Solovay Kitaev discrete compilation algorithm.

    This class is called recursively by the transpiler pass, which is why it is separeted.
    See :class:`~qiskit.transpiler.passes.SolovayKitaev` for more information.
    """

    def __init__(
        self, basis_gates: list[Gate | str] | None = None, depth: int = 16, do_checks: bool = False
    ) -> None:
        """
        Args:
            basis_gates: A list of discrete (i.e., non-parameterized) standard gates.
                Defaults to ``["h", "t", "tdg"]``.
            depth: The number of basis gate combinations to consider in the basis set. This
                determines how fast (and if) the algorithm converges and should be chosen
                sufficiently high.
            do_checks: If ``True``, perform intermediate steps checking whether the matrices
                are of expected form.
        """
        # allow read-only access to depth and basis gates
        self._depth = depth
        self._basis_gates = basis_gates

        if basis_gates is None:
            basis_gates = [HGate(), TGate(), TdgGate()]
        else:
            basis_gates = normalize_gates(basis_gates)

        self._sk = RustSolovayKitaevCompiler(basis_gates, depth, do_checks)

    @property
    def depth(self) -> int:
        """The depth in the basic approximations."""
        return self._depth

    @property
    def basis_gates(self) -> list[Gate]:
        """The basis gates in the basic approximations."""
        return self._basis_gates

    def synthesize_matrix(self, matrix: np.ndarray, recursion_degree: int) -> QuantumCircuit:
        """Run the Solovay-Kitaev algorithm on a :math:`U(2)` input matrix.

        For better accuracy, it is suggested to use :meth:`synthesize`, which provides the
        :class:`.Gate` to decompose and allows Qiskit to internally create a high-accuracy
        representation.

        Args:
            matrix: A 2x2 complex matrix representing a 1-qubit gate.
            recursion_degree: The recursion degree of the algorithm.

        Returns:
            The circuit implementing the approximation.
        """
        if matrix.shape != (2, 2):
            raise ValueError(f"Matrix must be U(2), but shape is {matrix.shape}.")

        data = self._sk.synthesize_matrix(matrix, recursion_degree)
        circuit = QuantumCircuit._from_circuit_data(data, add_regs=True)

        return circuit

    def synthesize(self, gate: Gate, recursion_degree: int) -> QuantumCircuit:
        """Run the Solovay-Kitaev algorithm on a standard gate.

        Args:
            gate: The standard gate to approximate.
            recursion_degree: The recursion degree of the algorithm.

        Returns:
            The circuit implementing the approximation.
        """
        data = self._sk.synthesize(gate, recursion_degree)
        circuit = QuantumCircuit._from_circuit_data(data, add_regs=True)

        return circuit

    def find_basic_approximation(self, gate: Gate) -> QuantumCircuit:
        """Query the basic approximation for a :class:`.Gate`.

        Args:
            gate: The gate to find the approximation of. To query the approximation of
                an arbitrary :math:`U(2)` matrix, wrap the matrix inside a
                :class:`.UnitaryGate`.

        Returns:
            The sequence in the set of basic approximations closest to the input.
        """
        data = self._sk.find_basic_approximation(gate)
        return QuantumCircuit._from_circuit_data(data, add_regs=True)


def normalize_gates(gates: list[Gate | str]) -> list[Gate]:
    """Normalize a list[Gate | str] into list[Gate]."""
    name_to_gate = get_standard_gate_name_mapping()

    def normalize(gate: Gate | str) -> Gate:
        if isinstance(gate, Gate):
            return gate
        if gate in name_to_gate:
            return name_to_gate[gate]
        raise ValueError(f"Unsupported gate: {gate}")

    return list(map(normalize, gates))


class SolovayKitaevDecomposition:
    """The Solovay Kitaev discrete decomposition algorithm.

    This class is called recursively by the transpiler pass, which is why it is separated.
    See :class:`qiskit.transpiler.passes.SolovayKitaev` for more information.
    """

    def __init__(
        self,
        basic_approximations: str | dict[str, np.ndarray] | list[GateSequence] | None = None,
        basis_gates: list[str] | None = None,
        depth: int = 16,
        do_checks: bool = False,
    ) -> None:
        """
        Args:
            basic_approximations: A specification of the basic SO(3) approximations in terms
                of discrete gates. At each iteration this algorithm, the remaining error is
                approximated with the closest sequence of gates in this set.
                If a ``str``, this specifies a ``.npy`` filename from which to load the
                approximation. If a ``dict``, then this contains
                ``{gates: effective_SO3_matrix}`` pairs,
                e.g. ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``.
                If a list, this contains the same information as the dict, but already converted to
                :class:`.GateSequence` objects, which contain the SO(3) matrix and gates.

                Either this parameter, or ``basis_gates`` and ``depth`` can be specified.
            basis_gates: A list of discrete (i.e., non-parameterized) standard gates.
                Defaults to ``["h", "t", "tdg"]``.
            depth: The number of basis gate combinations to consider in the basis set. This
                determines how fast (and if) the algorithm converges and should be chosen
                sufficiently high.
            do_checks: If ``True``, perform intermediate steps checking whether the matrices
                are of expected form.
        """
        if basic_approximations is None:
            if basis_gates is not None:
                basis_gates = [_1q_gates[name] for name in basis_gates]
            self._sk = RustSolovayKitaevCompiler(basis_gates, depth, None, do_checks)

        elif basis_gates is not None:
            raise ValueError(
                "Either basic_approximations or basis_gates + depth can be specified, not both."
            )

        else:
            # Fast Rust path to load the file
            if isinstance(basic_approximations, str) and basic_approximations[~4:] != ".npy":
                self._sk = RustSolovayKitaevCompiler.from_basic_approximations(
                    basic_approximations, True
                )
            else:
                sequences = self.load_basic_approximations(basic_approximations)
                self._sk = RustSolovayKitaevCompiler.from_sequences(sequences, True)

        self._depth = depth
        self._do_checks = do_checks
        self._basis_gates = basis_gates

    @property
    def depth(self) -> int:
        """The maximum gate depth of the basic approximations."""
        return self._depth

    @property
    def do_checks(self) -> bool:
        """Whether to perform runtime checks on the internal data."""
        return self._do_checks

    @property
    def basis_gates(self) -> list[str] | None:
        """The basis gate set of the basic approximations.

        If ``None``, defaults to ``["h", "t", "tdg"]``.
        """
        return self._basis_gates

    @staticmethod
    @deprecate_func(
        since="2.1",
        additional_msg="Loading basic approximations is more performant via "
        "SolovayKitaevDecomposition(<filename>), where the file is generated via "
        "SolovayKitaevDecomposition.save_basic_approximations().",
        pending=True,
    )
    def load_basic_approximations(data: list | str | dict) -> list[GateSequence]:
        """Load basic approximations.

        Args:
            data: If a string, specifies the path to the file from where to load the data.
                If a dictionary, directly specifies the decompositions as ``{gates: matrix}``
                or ``{gates: (matrix, global_phase)}``. There, ``gates`` are the names of the gates
                producing the SO(3) matrix ``matrix``, e.g.
                ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``
                and the ``global_phase`` can be given to account for a global phase difference
                between the U(2) matrix of the quantum gates and the stored SO(3) matrix.
                If not given, the ``global_phase`` will be assumed to be 0.

        Returns:
            A list of basic approximations as type ``GateSequence``.

        Raises:
            ValueError: If the number of gate combinations and associated matrices does not match.
        """
        # is already a list of GateSequences
        if isinstance(data, list):
            return data

        # if a file, load the dictionary
        if isinstance(data, str):
            data = np.load(data, allow_pickle=True).item()

        sequences = []
        for gatestring, matrix_and_phase in data.items():
            if isinstance(matrix_and_phase, tuple):
                matrix, global_phase = matrix_and_phase
            else:
                matrix, global_phase = matrix_and_phase, 0

            gates = [_1q_gates[element] for element in gatestring.split()]
            sequence = GateSequence.from_gates_and_matrix(gates, matrix, global_phase)
            sequences.append(sequence)

        return sequences

    def save_basic_approximations(self, filename: str):
        """Save the basic approximations into a file."""
        self._sk.save_basic_approximations(filename)

    def run(
        self,
        gate_matrix: np.ndarray | Gate,
        recursion_degree: int,
        return_dag: bool = False,
        check_input: bool = True,
    ) -> QuantumCircuit | DAGCircuit:
        r"""Run the algorithm.

        Args:
            gate_matrix: The 2x2 matrix representing the gate. This matrix has to be SU(2)
                up to global phase.
            recursion_degree: The recursion degree, called :math:`n` in the paper.
            return_dag: If ``True`` return a :class:`.DAGCircuit`, else a :class:`.QuantumCircuit`.
            check_input: If ``True`` check that the input matrix is valid for the decomposition.

        Returns:
            A one-qubit circuit approximating the ``gate_matrix`` in the specified discrete basis.
        """
        if isinstance(gate_matrix, Gate):
            if hasattr(gate_matrix, "_standard_gate"):
                data = self._sk.synthesize(gate_matrix, recursion_degree)
            else:
                data = self._sk.synthesize_matrix(gate_matrix.to_matrix(), recursion_degree)
        else:
            data = self._sk.synthesize_matrix(gate_matrix, recursion_degree)

        circuit = QuantumCircuit._from_circuit_data(data, add_regs=True)

        if return_dag:
            from qiskit.converters import circuit_to_dag  # pylint: disable=cyclic-import

            return circuit_to_dag(circuit)

        return circuit

    def query_basic_approximation(self, gate: np.ndarray | Gate) -> QuantumCircuit:
        """Query a basic approximation of a matrix."""
        if isinstance(gate, Gate):
            if hasattr(gate, "_standard_gate"):
                data = self._sk.query_basic_approximation(gate)
                return QuantumCircuit._from_circuit_data(data, add_regs=True)

            gate = gate.to_matrix()

        data = self._sk.query_basic_approximation_matrix(gate)
        circuit = QuantumCircuit._from_circuit_data(data, add_regs=True)
        return circuit

    def find_basic_approximation(self, sequence: GateSequence) -> GateSequence:
        """Find ``GateSequence`` in ``self._basic_approximations`` that approximates ``sequence``.

        Args:
            sequence: ``GateSequence`` to find the approximation to.

        Returns:
            ``GateSequence`` in ``self._basic_approximations`` that approximates ``sequence``.
        """
        return self._sk.find_basic_approximation(sequence)
