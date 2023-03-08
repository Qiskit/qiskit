# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for a stabilizer state preparation circuit.
"""
# pylint: disable=invalid-name

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import StabilizerState
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.clifford.clifford_decompose_layers import (
    synth_clifford_layers,
    _default_cz_synth_func,
)


def synth_stabilizer_layers(
    stab,
    cz_synth_func=_default_cz_synth_func,
    cz_func_reverse_qubits=False,
    validate=False,
):
    """Synthesis of a stabilizer state into layers.

    It provides a similar decomposition to the synthesis described in Lemma 8 of Bravyi and Maslov,
    without the initial Hadamard-free sub-circuit which do not affect the stabilizer state.

    For example, a 5-qubit Clifford circuit is decomposed into the following layers:

    .. parsed-literal::
             ┌─────┐┌─────┐┌─────┐┌─────┐┌────────┐
        q_0: ┤0    ├┤0    ├┤0    ├┤0    ├┤0       ├
             │     ││     ││     ││     ││        │
        q_1: ┤1    ├┤1    ├┤1    ├┤1    ├┤1       ├
             │     ││     ││     ││     ││        │
        q_2: ┤2 H2 ├┤2 S1 ├┤2 CZ ├┤2 H1 ├┤2 Pauli ├
             │     ││     ││     ││     ││        │
        q_3: ┤3    ├┤3    ├┤3    ├┤3    ├┤3       ├
             │     ││     ││     ││     ││        │
        q_4: ┤4    ├┤4    ├┤4    ├┤4    ├┤4       ├
             └─────┘└─────┘└─────┘└─────┘└────────┘

    Args:
        stab (StabilizerState): a stabilizer state.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.
            It gets as input a boolean symmetric matrix, and outputs a QuantumCircuit.
        validate (Boolean): if True, validates the synthesis process.
        cz_func_reverse_qubits (Boolean): True only if cz_synth_func is synth_cz_depth_line_mr,
            since this function returns a circuit that reverts the order of qubits.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    Raises:
        QiskitError: if the input is not a StabilizerState.

    Reference:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """

    if not isinstance(stab, StabilizerState):
        raise QiskitError("The input is not a StabilizerState.")

    cliff = stab.clifford
    num_qubits = cliff.num_qubits
    qubit_list = list(range(num_qubits))

    circ = synth_clifford_layers(
        cliff,
        cz_synth_func=cz_synth_func,
        cz_func_reverse_qubits=cz_func_reverse_qubits,
        validate=validate,
    )
    H2_circ = circ.data[-5]
    S1_circ = circ.data[-4]
    CZ1_circ = circ.data[-3]
    H1_circ = circ.data[-2]
    pauli_circ = circ.data[-1]

    stab_circuit = QuantumCircuit(num_qubits)
    stab_circuit.append(H2_circ, qubit_list)
    stab_circuit.append(S1_circ, qubit_list)
    stab_circuit.append(CZ1_circ, qubit_list)
    stab_circuit.append(H1_circ, qubit_list)
    stab_circuit.append(pauli_circ, qubit_list)

    return stab_circuit


def synth_stabilizer_depth_lnn(stab):
    """Synthesis of an n-qubit stabilizer state for linear-nearest neighbour connectivity,
    in 2-qubit depth 2*n+2, and two distinct CX layers.

    Args:
        stab (StabilizerState): a stabilizer state.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    Reference:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
        2. Dmitri Maslov, Martin Roetteler,
           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
    """

    circ = synth_stabilizer_layers(
        stab,
        cz_synth_func=synth_cz_depth_line_mr,
        cz_func_reverse_qubits=True,
    )
    return circ
