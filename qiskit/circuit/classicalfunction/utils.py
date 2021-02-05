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

"""Internal utils for ClassicalFunction Compiler"""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate, \
    XGate, HGate, U3Gate
from qiskit.circuit.classicalfunction.exceptions import ClassicalFunctionCompilerError


def tweedledum2qiskit(tweedledum_circuit, name=None, qregs=None):
    """ Converts a `Tweedledum <https://github.com/boschmitt/tweedledum>`_
    circuit into a Qiskit circuit. A Tweedledum circuit is a
    dictionary with the following shape:
        {
        "num_qubits": 2,
        "gates": [{
            "gate": "X",
            "qubits": [1],
            "control_qubits": [0],
            "control_state": "1"
        }]
    Args:
        tweedledum_circuit (dict): Tweedledum circuit.
        name (str): Name for the resulting Qiskit circuit.
        qregs (list(QuantumRegister)): Optional. List of QuantumRegisters on which the
           circuit would operate. If not provided, it will create a flat register.

    Returns:
        QuantumCircuit: The Tweedledum circuit converted to a Qiskit circuit.

    Raises:
        ClassicalFunctionCompilerError: If there a gate in the Tweedledum circuit has no Qiskit
        equivalent.
    """
    gates = {'z': ZGate, 't': TGate, 's': SGate, 'tdg': TdgGate, 'sdg': SdgGate, 'u1': U1Gate,
             'x': XGate, 'h': HGate, 'u3': U3Gate}
    if qregs:
        circuit = QuantumCircuit(*qregs, name=name)
    else:
        circuit = QuantumCircuit(tweedledum_circuit['num_qubits'], name=name)
    for gate in tweedledum_circuit['gates']:
        basegate = gates.get(gate['gate'].lower())
        if basegate is None:
            raise ClassicalFunctionCompilerError('The Tweedledum gate %s has no Qiskit equivalent'
                                                 % gate['gate'])

        ctrl_qubits = gate.get('control_qubits', [])
        trgt_qubits = gate.get('qubits', [])

        if ctrl_qubits:
            gate = basegate().control(len(ctrl_qubits), ctrl_state=gate.get('control_state'))
        else:
            gate = basegate()
        circuit.append(gate, ctrl_qubits + trgt_qubits)
    return circuit
