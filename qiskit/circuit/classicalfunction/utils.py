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

from qiskit.utils.optionals import HAS_TWEEDLEDUM

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import (
    HGate,
    SGate,
    SdgGate,
    SwapGate,
    TGate,
    TdgGate,
    XGate,
    YGate,
    ZGate,
)

_QISKIT_OPS = {
    "std.h": HGate,
    "std.s": SGate,
    "std.sdg": SdgGate,
    "std.swap": SwapGate,
    "std.t": TGate,
    "std.tdg": TdgGate,
    "std.x": XGate,
    "std.y": YGate,
    "std.z": ZGate,
}


@HAS_TWEEDLEDUM.require_in_call
def _convert_tweedledum_operator(op):
    base_gate = _QISKIT_OPS.get(op.kind())
    if base_gate is None:
        if op.kind() == "py_operator":
            return op.py_op()
        else:
            raise RuntimeError("Unrecognized operator: %s" % op.kind())

    # TODO: need to deal with cbits too!
    if op.num_controls() > 0:
        from tweedledum.ir import Qubit  # pylint: disable=import-error

        qubits = op.qubits()
        ctrl_state = ""
        for qubit in qubits[: op.num_controls()]:
            ctrl_state += f"{int(qubit.polarity() == Qubit.Polarity.positive)}"
        return base_gate().control(len(ctrl_state), ctrl_state=ctrl_state[::-1])
    return base_gate()


@HAS_TWEEDLEDUM.require_in_call
def tweedledum2qiskit(tweedledum_circuit, name=None, qregs=None):
    """Converts a `Tweedledum <https://github.com/boschmitt/tweedledum>`_
    circuit into a Qiskit circuit.
    Args:
        tweedledum_circuit (tweedledum.ir.Circuit): Tweedledum circuit.
        name (str): Name for the resulting Qiskit circuit.
        qregs (list(QuantumRegister)): Optional. List of QuantumRegisters on which the
           circuit would operate. If not provided, it will create a flat register.

    Returns:
        QuantumCircuit: The Tweedledum circuit converted to a Qiskit circuit.

    Raises:
        ClassicalFunctionCompilerError: If a gate in the Tweedledum circuit has no Qiskit equivalent.
    """
    if qregs:
        qiskit_qc = QuantumCircuit(*qregs, name=name)
    else:
        qiskit_qc = QuantumCircuit(tweedledum_circuit.num_qubits(), name=name)

    from tweedledum.passes import parity_decomp  # pylint: disable=import-error

    for instruction in parity_decomp(tweedledum_circuit):
        gate = _convert_tweedledum_operator(instruction)
        qubits = [qubit.uid() for qubit in instruction.qubits()]
        qiskit_qc.append(gate, qubits)
    return qiskit_qc
