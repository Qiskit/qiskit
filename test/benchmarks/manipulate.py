# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member,invalid-name,missing-docstring,no-name-in-module
# pylint: disable=attribute-defined-outside-init,unsubscriptable-object
# pylint: disable=unused-wildcard-import,wildcard-import,undefined-variable

import os
import numpy as np

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit import CircuitInstruction, Qubit, library
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import PropertySet
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from .utils import multi_control_circuit

GATES = {
    "id": library.IGate(),
    "x": library.XGate(),
    "y": library.YGate(),
    "z": library.ZGate(),
    "cx": library.CXGate(),
    "cz": library.CZGate(),
}

TWIRLING_SETS_NAMES = {
    "cx": [
        ["id", "z", "z", "z"],
        ["id", "x", "id", "x"],
        ["id", "y", "z", "y"],
        ["id", "id", "id", "id"],
        ["z", "x", "z", "x"],
        ["z", "y", "id", "y"],
        ["z", "id", "z", "id"],
        ["z", "z", "id", "z"],
        ["x", "y", "y", "z"],
        ["x", "id", "x", "x"],
        ["x", "z", "y", "y"],
        ["x", "x", "x", "id"],
        ["y", "id", "y", "x"],
        ["y", "z", "x", "y"],
        ["y", "x", "y", "id"],
        ["y", "y", "x", "z"],
    ],
    "cz": [
        ["id", "z", "id", "z"],
        ["id", "x", "z", "x"],
        ["id", "y", "z", "y"],
        ["id", "id", "id", "id"],
        ["z", "x", "id", "x"],
        ["z", "y", "id", "y"],
        ["z", "id", "z", "id"],
        ["z", "z", "z", "z"],
        ["x", "y", "y", "x"],
        ["x", "id", "x", "z"],
        ["x", "z", "x", "id"],
        ["x", "x", "y", "y"],
        ["y", "id", "y", "z"],
        ["y", "z", "y", "id"],
        ["y", "x", "x", "y"],
        ["y", "y", "x", "x"],
    ],
}
TWIRLING_SETS = {
    key: [[GATES[name] for name in twirl] for twirl in twirls]
    for key, twirls in TWIRLING_SETS_NAMES.items()
}


def _dag_from_twirl(gate_2q, twirl):
    dag = DAGCircuit()
    # or use QuantumRegister - doesn't matter
    qubits = (Qubit(), Qubit())
    dag.add_qubits(qubits)
    dag.apply_operation_back(twirl[0], (qubits[0],), (), check=False)
    dag.apply_operation_back(twirl[1], (qubits[1],), (), check=False)
    dag.apply_operation_back(gate_2q, qubits, (), check=False)
    dag.apply_operation_back(twirl[2], (qubits[0],), (), check=False)
    dag.apply_operation_back(twirl[3], (qubits[1],), (), check=False)
    return dag


def circuit_twirl(qc, twirled_gate="cx", seed=None):
    rng = np.random.default_rng(seed)
    twirl_set = TWIRLING_SETS.get(twirled_gate, [])

    out = qc.copy_empty_like()
    for instruction in qc.data:
        if instruction.operation.name != twirled_gate:
            out._append(instruction)
        else:
            # We could also scan through `qc` outside the loop to know how many
            # twirled gates we'll be dealing with, and RNG the integers ahead of
            # time - that'll be faster depending on what percentage of gates are
            # twirled, and how much the Numpy overhead is.
            twirls = twirl_set[rng.integers(len(twirl_set))]
            control, target = instruction.qubits
            out._append(CircuitInstruction(twirls[0], (control,), ()))
            out._append(CircuitInstruction(twirls[1], (target,), ()))
            out._append(instruction)
            out._append(CircuitInstruction(twirls[2], (control,), ()))
            out._append(CircuitInstruction(twirls[3], (target,), ()))
    return out


def dag_twirl(dag, twirled_gate="cx", seed=None):
    # This mutates `dag` in place.
    rng = np.random.default_rng(seed)
    twirl_set = TWIRLING_DAGS.get(twirled_gate, [])
    twirled_gate_op = GATES[twirled_gate].base_class

    to_twirl = dag.op_nodes(twirled_gate_op)
    twirl_indices = rng.integers(len(twirl_set), size=(len(to_twirl),))

    for index, op_node in zip(twirl_indices, to_twirl):
        dag.substitute_node_with_dag(op_node, twirl_set[index])
    return dag


TWIRLING_DAGS = {
    key: [_dag_from_twirl(GATES[key], twirl) for twirl in twirls]
    for key, twirls in TWIRLING_SETS.items()
}


class TestCircuitManipulate:
    def setup(self):
        qasm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qasm")
        self.qft_qasm = os.path.join(qasm_dir, "dtc_100_cx_12345.qasm")
        self.qft_qc = QuantumCircuit.from_qasm_file(self.qft_qasm)
        self.qv_qasm = os.path.join(qasm_dir, "qv_N100_12345.qasm")
        self.qv_qc = QuantumCircuit.from_qasm_file(self.qv_qasm)
        self.dtc_qasm = os.path.join(qasm_dir, "dtc_100_cx_12345.qasm")
        self.dtc_qc = QuantumCircuit.from_qasm_file(self.dtc_qasm)
        self.translate = generate_preset_pass_manager(1, basis_gates=["rx", "ry", "rz", "cz"])

    def time_DTC100_twirling(self):
        """Perform Pauli-twirling on a 100Q QV
        circuit
        """
        out = circuit_twirl(self.dtc_qc)
        return out

    def time_multi_control_decompose(self):
        """Decompose a multi-control gate into the
        basis [rx, ry, rz, cz]
        """
        circ = multi_control_circuit(16)
        self.translate.property_set = PropertySet()
        out = self.translate.run(circ)
        return out

    def time_QV100_basis_change(self):
        """Change a QV100 circuit basis from [rx, ry, rz, cx]
        to [sx, x, rz, cz]
        """
        self.translate.property_set = PropertySet()
        out = self.translate.run(self.qv_qc)
        return out

    def time_DTC100_twirling_dag(self):
        """Perform Pauli-twirling on a 100Q QV
        circuit
        """
        self.translate.property_set = PropertySet()
        out = self.translate.run(self.qv_qc)
        return circuit_to_dag(out)
