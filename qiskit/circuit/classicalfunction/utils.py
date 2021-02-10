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

"""Internal utils for Classical Function Compiler"""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate, \
    XGate, HGate, U3Gate
from qiskit.circuit.classicalfunction.exceptions import ClassicalFunctionCompilerError

from tweedledum.ir import WireRef
from tweedledum.passes import parity_decomp

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import (HGate, SGate, SdgGate,
    SwapGate, TGate, TdgGate, XGate, YGate, ZGate)

_to_qiskit_op = {
    'std.h': HGate, 'std.s': SGate, 'std.sdg': SdgGate, 'std.swap': SwapGate,
    'std.t': TGate, 'std.tdg': TdgGate, 'std.x': XGate, 'std.y': YGate,
    'std.z': ZGate
}

def _convert_tweedledum_operator(op):
    base_gate = _to_qiskit_op.get(op.kind())
    if base_gate == None:
        if op.kind() == 'py_operator':
            return op.py_op()
        else:
            raise RuntimeError('Unrecognized operator: %s' % op.kind())

    # TODO: need to deal with cbits too!
    if op.num_controls() > 0:
        qubits = op.qubits()
        ctrl_state = ''
        for qubit in qubits[:op.num_controls()]:
            ctrl_state += '{}'.format(int(qubit.polarity() == WireRef.Polarity.positive))
        return base_gate().control(len(ctrl_state), ctrl_state=ctrl_state[::-1])
    return base_gate()

    # TODO:
    # elif instruction.kind() == 'std.p':
    #     return PhaseGate()
    # elif instruction.kind() == 'std.rx':
    #     return RZGate()
    # elif instruction.kind() == 'std.ry':
    #     return RZGate()
    # elif instruction.kind() == 'std.rz':
    #     return RZGate()

def tweedledum2qiskit(tweedledum_circuit, name=None, qregs=None):
    """ Converts a `Tweedledum <https://github.com/boschmitt/tweedledum>`_
    circuit into a Qiskit circuit.
    Args:
        tweedledum_circuit (tweedledum.ir.Circuit): Tweedledum circuit.
        name (str): Name for the resulting Qiskit circuit.
        qregs (list(QuantumRegister)): Optional. List of QuantumRegisters on which the
           circuit would operate. If not provided, it will create a flat register.

    Returns:
        QuantumCircuit: The Tweedledum circuit converted to a Qiskit circuit.

    Raises:
        ClassicalFunctionCompilerError: If there a gate in the Tweedledum circuit has no Qiskit
        equivalent.
    """
    qiskit_qc = QuantumCircuit(tweedledum_circuit.num_qubits())
    for instruction in parity_decomp(tweedledum_circuit):
        gate = _convert_tweedledum_operator(instruction)
        qubits = [qubit.uid() for qubit in instruction.qubits()]
        qiskit_qc.append(gate, qubits)
    return qiskit_qc
