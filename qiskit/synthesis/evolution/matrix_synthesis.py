# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exact synthesis of operator evolution via (exponentially expensive) matrix exponentiation."""

from qiskit.circuit.quantumcircuit import QuantumCircuit

from .evolution_synthesis import EvolutionSynthesis


class MatrixExponential(EvolutionSynthesis):
    r"""Exact operator evolution via matrix exponentiation and unitary synthesis.

    This class synthesis the exponential of operators by calculating their exponentially-sized
    matrix representation and using exact matrix exponentiation followed by unitary synthesis
    to obtain a circuit. This process is not scalable and serves as comparison or benchmark
    for small systems.
    """

    def synthesize(self, evolution):
        from scipy.linalg import expm

        # get operators and time to evolve
        operators = evolution.operator
        time = evolution.time

        # construct the evolution circuit
        evolution_circuit = QuantumCircuit(operators[0].num_qubits)

        if not isinstance(operators, list):
            matrix = operators.to_matrix()
        else:
            matrix = sum(op.to_matrix() for op in operators)

        exp = expm(-1j * time * matrix)

        evolution_circuit.unitary(exp, evolution_circuit.qubits)

        return evolution_circuit
