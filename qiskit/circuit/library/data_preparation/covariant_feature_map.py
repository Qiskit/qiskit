# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Covariant feature map circuit."""

from typing import Union, Sequence, Optional

from qiskit.circuit import QuantumCircuit, ParameterVector


class CovariantFeatureMap(QuantumCircuit):
    """Covariant feature map circuit.

    On 3 qubits and a linear entanglement,  the circuit is represented by:

    .. parsed-literal::


         ┌──────────┐       ░ ┌─────────────┐┌─────────────┐
    q_0: ┤ Ry(θ[0]) ├─■─────░─┤ Rz(-2*x[1]) ├┤ Rx(-2*x[0]) ├
         ├──────────┤ │     ░ ├─────────────┤├─────────────┤
    q_1: ┤ Ry(θ[1]) ├─■──■──░─┤ Rz(-2*x[3]) ├┤ Rx(-2*x[2]) ├
         ├──────────┤    │  ░ ├─────────────┤├─────────────┤
    q_2: ┤ Ry(θ[2]) ├────■──░─┤ Rz(-2*x[5]) ├┤ Rx(-2*x[4]) ├
         └──────────┘       ░ └─────────────┘└─────────────┘

    where θ is a vector of parameters to optimize, and x is a
    vector of parameters specified by the feature data for a given sample.

    **Reference:**

    [1] Jennifer R. Glick et al., Covariant quantum kernels for data with group
    structure, 2022.
    `arXiv:2105.03406 <https://arxiv.org/pdf/2105.03406.pdf>`_
    """

    def __init__(
        self,
        feature_dimension: int,
        entanglement: Optional[Union[Sequence[Sequence[int]], str]] = "linear",
        single_training_parameter: bool = False,
        training_parameter_prefix: str = "θ",
        feature_parameter_prefix: str = "x",
    ) -> None:
        """Create Covariant feature map circuit.

        Args:
            feature_dimension: Number of features.
            entanglement: Specifies the entanglement structure. Entanglement may be specified
                by one of the following strings:
                    - linear
                    - reverse_linear
                    - circular
                    - pairwise
                    - full
                Alternatively, entanglement may be specified as a sequence of length-2 sequences
                containing indices between qubits which should be entangled with :class:`.CXGate`\ s.
                For example, ``entanglement = [(3, 2), (2, 1), (1, 0)]`` specifies reverse-linear
                entanglement on a 4-qubit circuit.
            single_training_parameter: If ``True``, each qubit will share a single
                training parameter. If False, the initial y-axis rotation for each qubit
                will be optimized individually.
            training_parameter_prefix: The name of the parameter vector used to hold
                the training parameters. Defaults to ``θ``.
            feature_parameter_prefix: The name of the parameter vector used to hold
                the parameters specified by the input feature data for a given sample.
                Defaults to ``x``.

        Raises:
            ValueError: Feature dimension must be a multiple of 2.
            ValueError: Unrecognized entanglement structure.
        """
        if feature_dimension % 2 != 0:
            raise ValueError("Feature dimension must be a multiple of two.")
        num_qubits = int(feature_dimension / 2)

        if isinstance(entanglement, str):
            entanglement_map = self._str_to_entanglement_map(entanglement, num_qubits)
        else:
            entanglement_map = entanglement

        circuit = QuantumCircuit(num_qubits, name="CovariantFeatureMap")

        # Vector of data parameters
        feature_params = ParameterVector("x", feature_dimension)

        # Use a single parameter for each initial Y rotation
        if single_training_parameter:
            training_params = ParameterVector("θ", 1)
            for i in range(num_qubits):
                circuit.ry(training_params[0], circuit.qubits[i])

        # Train each qubit's initial rotation independently
        else:
            training_params = ParameterVector("θ", num_qubits)
            for i in range(num_qubits):
                circuit.ry(training_params[i], circuit.qubits[i])

        # Create the entangling layer
        for source, target in entanglement_map:
            circuit.cz(circuit.qubits[source], circuit.qubits[target])

        # Create a circuit representation of the data group
        for i in range(num_qubits):
            circuit.rz(-2 * feature_params[2 * i + 1], circuit.qubits[i])
            circuit.rx(-2 * feature_params[2 * i], circuit.qubits[i])

        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

    @staticmethod
    def _str_to_entanglement_map(entanglement: str, num_qubits: int) -> list[tuple[int, int]]:
        if entanglement == "linear":
            return [[i, i + 1] for i in range(num_qubits - 1)]
        elif entanglement == "reverse_linear":
            return [[i, i - 1] for i in range(num_qubits - 1, 0, -1)]
        elif entanglement == "circular":
            return [[i, i + 1] for i in range(-1, num_qubits - 1, 1)]
        elif entanglement == "pairwise":
            return [(i, i + 1) for i in range(0, num_qubits - 1, 2)] + [
                (i, i + 1) for i in range(1, num_qubits - 2, 2)
            ]
        elif entanglement == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        else:
            raise ValueError(f"Unknown entanglement structure: {entanglement}")
