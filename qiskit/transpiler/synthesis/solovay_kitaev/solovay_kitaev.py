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

from typing import Union, Optional, Dict, List

import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError

from .utils import GateSequence, commutator_decompose
from .generate_basis_approximations import generate_basic_approximations, _1q_gates, _1q_inverses


class SolovayKitaev:
    """The Solovay Kitaev discrete decomposition algorithm.

    See :class:`~qiskit.transpiler.passes.SolovayKitaevDecomposition` for more information.
    """

    def __init__(
        self,
        basic_approximations: Optional[
            Union[str, Dict[str, np.ndarray], List[GateSequence]]
        ] = None,
    ) -> None:
        """

        .. note::

            If ``basis_gates`` and ``depth`` are not passed, the basic approximations can be
            generated with the ``generate_basic_approximations`` method and loaded into the
            class via ``load_basic_approximations``. Since in practice, large basic approximations
            are required we suggest to generate a sufficiently large set once and always load
            the approximations afterwards.

        Args:
            basis_gates: The basis gates used in the basic approximations.
            depth: The maximum depth of the basic approximations.
        """
        if basic_approximations is None:
            # generate a default basic approximation
            basic_approximations = generate_basic_approximations(
                basis_gates=["h", "t", "tdg"], depth=10
            )

        self.basic_approximations = self.load_basic_approximations(basic_approximations)

    def load_basic_approximations(self, data: Union[dict, str]) -> List[GateSequence]:
        """Load basic approximations.

        Args:
            data: If a string, specifies the path to the file from where to load the data.
                If a dictionary, directly specifies the decompositions as ``{gates: matrix}``.
                There ``gates`` are the names of the gates producing the SO(3) matrix ``matrix``,
                e.g. ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``.

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
            data = np.load(data, allow_pickle=True)

        sequences = []
        for gatestring, matrix in data.items():
            sequence = GateSequence()
            sequence.gates = [_1q_gates[element] for element in gatestring.split()]
            sequence.product = np.asarray(matrix)
            sequences.append(sequence)

        return sequences

    def run(self, gate_matrix: np.ndarray, recursion_degree: int) -> QuantumCircuit:
        r"""Run the algorithm.

        Args:
            gate_matrix: The 2x2 matrix representing the gate. Does not need to be SU(2).
            recursion_degree: The recursion degree, called :math:`n` in the paper.

        Returns:
            A one-qubit circuit approximating the ``gate_matrix`` in the specified discrete basis.
        """
        # make input matrix SU(2) and get the according global phase
        z = 1 / np.sqrt(np.linalg.det(gate_matrix))
        gate_matrix_su2 = GateSequence.from_matrix(z * gate_matrix)
        global_phase = np.arctan2(np.imag(z), np.real(z))

        # get the decompositon as GateSequence type
        decomposition = self._recurse(gate_matrix_su2, recursion_degree)

        # simplify
        _remove_identities(decomposition)
        _remove_inverse_follows_gate(decomposition)

        # convert to a circuit and attach the right phases
        # TODO insert simplify again, but it seems to break the accuracy test
        circuit = decomposition.to_circuit()
        circuit.global_phase = decomposition.global_phase - global_phase

        return circuit

    def _recurse(self, sequence: GateSequence, n: int) -> GateSequence:
        """Performs ``n`` iterations of the Solovay-Kitaev algorithm on ``sequence``.

        Args:
            sequence: GateSequence to which the Solovay-Kitaev algorithm is applied.
            n: number of iterations that the algorithm needs to run.

        Returns:
            GateSequence that approximates ``sequence``.

        Raises:
            ValueError: if ``u`` does not represent an SO(3)-matrix.
        """
        if sequence.product.shape != (3, 3):
            raise ValueError("Shape of U must be (3, 3) but is", sequence.shape)

        if n == 0:
            return self.find_basic_approximation(sequence)

        u_n1 = self._recurse(sequence, n - 1)

        v_n, w_n = commutator_decompose(sequence.dot(u_n1.adjoint()).product)

        v_n1 = self._recurse(v_n, n - 1)
        w_n1 = self._recurse(w_n, n - 1)
        return v_n1.dot(w_n1).dot(v_n1.adjoint()).dot(w_n1.adjoint()).dot(u_n1)

    def find_basic_approximation(self, sequence: GateSequence) -> Gate:
        """Finds gate in ``self._basic_approximations`` that best represents ``sequence``.

        Args:
            sequence: The gate to find the approximation to.

        Returns:
            Gate in basic approximations that is closest to ``sequence``.
        """
        # TODO explore using a k-d tree here

        def key(x):
            return np.linalg.norm(np.subtract(x.product, sequence.product))

        best = min(self.basic_approximations, key=key)
        return best


class SolovayKitaevDecomposition(TransformationPass):
    r"""Approximately decompose 1q gates to a discrete basis using the Solovay-Kitaev algorithm.

    The Solovay-Kitaev theorem [1] states that any single qubit gate can be approximated to
    arbitrary precision by a set of fixed single-qubit gates, if the set generates a dense
    subset in :math:`SU(2)`. This is an important result, since it means that any single-qubit
    gate can be expressed in terms of a discrete, universal gate set that we know how to implement
    fault-tolerantly. Therefore, the Solovay-Kitaev algorithm allows us to take any
    non-fault tolerant circuit and rephrase it in a fault-tolerant manner.

    This implementation of the Solovay-Kitaev algorithm is based on [2].

    For example, the following circuit

    .. parsed-literal::

             ┌─────────┐
        q_0: ┤ RX(0.8) ├
             └─────────┘

    can be decomposed into

    .. parsed-literal::

        global phase: 7π/8
             ┌───┐┌───┐┌───┐
        q_0: ┤ H ├┤ T ├┤ H ├
             └───┘└───┘└───┘

    with an L2-error of approximately 0.01.


    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit.circuit import QuantumCircuit
            from qiskit.circuit.library import TGate, HGate, TdgGate
            from qiskit.converters import circuit_to_dag, dag_to_circuit
            from qiskit.transpiler.passes import SolovayKitaevDecomposition
            from qiskit.quantum_info import Operator

            circuit = QuantumCircuit(1)
            circuit.rx(0.8, 0)
            dag = circuit_to_dag(circuit)

            print('Original circuit:')
            print(circuit.draw())

            basis_gates = [TGate(), TdgGate(), HGate()]
            skd = SolovayKitaevDecomposition(recursion_degree=2, basis_gates=basis_gates)

            discretized = dag_to_circuit(skd.run(dag))

            print('Discretized circuit:')
            print(discretized.draw())

            print('Error:', np.linalg.norm(Operator(circuit).data - Operator(discretized).data))


    References:

        [1]: Kitaev, A Yu (1997). Quantum computations: algorithms and error correction.
             Russian Mathematical Surveys. 52 (6): 1191–1249.
             `Online <https://iopscience.iop.org/article/10.1070/RM1997v052n06ABEH002155>`_.

        [2]: Dawson, Christopher M.; Nielsen, Michael A. (2005) The Solovay-Kitaev Algorithm.
             `arXiv:quant-ph/0505030 <https://arxiv.org/abs/quant-ph/0505030>`_


    """

    def __init__(
        self,
        recursion_degree: int = 3,
        basic_approximations: Optional[Union[str, Dict[str, np.ndarray]]] = None,
    ) -> None:
        """
        Args:
            recursion_degree: The recursion depth for the Solovay-Kitaev algorithm.
                A larger recursion depth increases the accuracy and length of the
                decomposition.
            basic_approximations: The basic approximations for the finding the best discrete
                decomposition at the root of the recursion. If a string, it specifies the ``.npy``
                file to load the approximations from. If a dictionary, it contains
                ``{label: SO(3)-matrix}`` pairs. If None, a default based on the H, T and Tdg gates
                up to combinations of depth 10 is generated.
        """
        super().__init__()
        self.recursion_degree = recursion_degree
        self._sk = SolovayKitaev(basic_approximations)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the SolovayKitaevDecomposition pass on `dag`.

        Args:
            dag: The input dag.

        Returns:
            Output dag with 1q gates synthesized in the discrete target basis.

        Raises:
            TranspilerError: if a gates does not have to_matrix
        """
        for node in dag.op_nodes():
            if not node.op.num_qubits == 1:
                continue  # ignore all non-single qubit gates

            if not hasattr(node.op, "to_matrix"):
                raise TranspilerError(
                    "SolovayKitaevDecomposition does not support gate without "
                    f"to_matrix method: {node.op.name}"
                )

            matrix = node.op.to_matrix()

            # call solovay kitaev
            approximation = self._sk.run(matrix, self.recursion_degree)

            # convert to a dag and replace the gate by the approximation
            substitute = circuit_to_dag(approximation)
            dag.substitute_node_with_dag(node, substitute)

        return dag


def _remove_inverse_follows_gate(sequence):
    index = 0
    while index < len(sequence.gates) - 1:
        curr_gate = sequence.gates[index]
        next_gate = sequence.gates[index + 1]
        if curr_gate.name in _1q_inverses.keys():
            remove = _1q_inverses[curr_gate.name] == next_gate.name
        else:
            remove = curr_gate.inverse() == next_gate

        if remove:
            # remove gates at index and index + 1
            sequence.remove_cancelling_pair([index, index + 1])
            # take a step back to see if we have uncovered a new pair, e.g.
            # [h, s, sdg, h] at index = 1 removes s, sdg but if we continue at index 1
            # we miss the uncovered [h, h] pair at indices 0 and 1
            if index > 0:
                index -= 1
        else:
            # next index
            index += 1


def _remove_identities(sequence):
    index = 0
    while index < len(sequence.gates):
        if sequence.gates[index].name == "id":
            sequence.gates.pop(index)
        else:
            index += 1
