# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Backend for the unroller that composes qasm into json file.

Author: Jay Gambetta and Andrew Cross

The input is a AST and a basis set and returns a json-like file

{
 "header": {
 "number_qubits": 2, // int
 "number_cbits": 2, // int
 "qubit_labels": [["q", 0], ["v", 0]], // list[list[string, int]]
 "cbits_labels": [["c", 2]], // list[list[string, int]]
 }
 "operations": // list[map]
    [
        {
            "name": , // required -- string
            "params": , // optional -- list[double]
            "qubits": , // optional -- list[int]
            "cbits": , //optional -- list[int]
            "conditional":  // optional -- map
                {
                    "type": , // string
                    "mask": , // big int
                    "val":  , // big int
                }
        },
    ]
}
"""
from qiskit.unroll import BackendException
from qiskit.unroll import UnrollerBackend


class JsonBackend(UnrollerBackend):
    """Backend for the unroller that makes a Json quantum circuit."""

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        The default basis is ["U", "CX"].
        """
        self.circuit = {}
        self.circuit['operations'] = []
        self.circuit['header'] = {}
        self._number_of_qubits = 0
        self._number_of_cbits = 0
        self._qubit_order = []
        self._cbit_order = []
        self._qubit_order_internal = {}
        self._cbit_order_internal = {}

        self.creg = None
        self.cval = None
        self.gates = {}
        if basis:
            self.basis = basis
        else:
            self.basis = []  # default, unroll to U, CX
        self.listen = True
        self.in_gate = ""
        self.printed_gates = []

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit.

        basis is a list of operation name strings.
        """
        self.basis = basis

    def version(self, version):
        """Print the version string.

        v is a version number.
        """
        pass

    def new_qreg(self, name, size):
        """Create a new quantum register.

        name = name of the register
        sz = size of the register
        """
        assert size >= 0, "invalid qreg size"

        for j in range(size):
            self._qubit_order.append([name, j])
            self._qubit_order_internal[(name, j)] = self._number_of_qubits + j
        self._number_of_qubits += size
        self.circuit['header']['number_of_qubits'] = self._number_of_qubits
        self.circuit['header']['qubit_labels'] = self._qubit_order

    def new_creg(self, name, size):
        """Create a new classical register.

        name = name of the register
        sz = size of the register
        """
        assert size >= 0, "invalid creg size"
        self._cbit_order.append([name, size])
        for j in range(size):
            self._cbit_order_internal[(name, j)] = self._number_of_cbits + j
        self._number_of_cbits += size
        self.circuit['header']['number_of_cbits'] = self._number_of_cbits
        self.circuit['header']['cbits_labels'] = self._cbit_order

    def define_gate(self, name, gatedata):
        """Define a new quantum gate.

        name is a string.
        gatedata is the AST node for the gate.
        """
        self.gates[name] = gatedata

    def u(self, arg, qubit):
        """Fundamental single-qubit gate.

        arg is 3-tuple of float parameters.
        qubit is (regname, idx) tuple.
        """
        if self.listen:
            if "U" not in self.basis:
                self.basis.append("U")
            qubit_indices = [self._qubit_order_internal.get(qubit)]
            self.circuit['operations'].append({
                'name': "U",
                'params': [arg[0], arg[1], arg[2]],
                'qubits': qubit_indices
                })
            self._add_condition()

    def _add_condition(self):
        """Check for a condition (self.creg) and add fields if necessary.

        Fields are added to the last operation in the circuit.
        """
        if self.creg is not None:
            mask = 0
            for cbit, index in self._cbit_order_internal.items():
                if cbit[0] == self.creg:
                    mask |= (1 << index)
                conditional = {
                    'type': "==",
                    'mask': mask,
                    'val': self.cval
                }
            self.circuit['operations'][-1]['conditional'] = conditional

    def cx(self, qubit0, qubit1):
        """Fundamental two-qubit gate.

        qubit0 is (regname, idx) tuple for the control qubit.
        qubit1 is (regname, idx) tuple for the target qubit.
        """
        if self.listen:
            if "CX" not in self.basis:
                self.basis.append("CX")
            qubit_indices = [self._qubit_order_internal.get(qubit0),
                             self._qubit_order_internal.get(qubit1)]
            self.circuit['operations'].append({
                'name': 'CX',
                'qubits': qubit_indices,
                })
            self._add_condition()

    def measure(self, qubit, cbit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        if "measure" not in self.basis:
            self.basis.append("measure")
        qubit_indices = [self._qubit_order_internal.get(qubit)]
        cbit_indices = [self._cbit_order_internal.get(cbit)]
        self.circuit['operations'].append({
            'name': 'measure',
            'qubits': qubit_indices,
            'cbits': cbit_indices
            })
        self._add_condition()

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        if self.listen:
            if "barrier" not in self.basis:
                self.basis.append("barrier")
            qubit_indices = []
            for qubitlist in qubitlists:
                for qubits in qubitlist:
                    qubit_indices.append(self._qubit_order_internal.get(qubits))
            self.circuit['operations'].append({
                'name': 'barrier',
                'qubits': qubit_indices,
                })
            # no conditions on barrier, even when it appears
            # in body of conditioned gate

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        if "reset" not in self.basis:
            self.basis.append("reset")
        qubit_indices = [self._qubit_order_internal.get(qubit)]
        self.circuit['operations'].append({
            'name': 'reset',
            'qubits': qubit_indices,
            })
        self._add_condition()

    def set_condition(self, creg, cval):
        """Attach a current condition.

        creg is a name string.
        cval is the integer value for the test.
        """
        self.creg = creg
        self.cval = cval

    def drop_condition(self):
        """Drop the current condition."""
        self.creg = None
        self.cval = None

    def start_gate(self, name, args, qubits):
        """Begin a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        if self.listen and name not in self.basis \
           and self.gates[name]["opaque"]:
            raise BackendException("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            self.in_gate = name
            self.listen = False
            qubit_indices = [self._qubit_order_internal.get(qubit)
                             for qubit in qubits]
            self.circuit['operations'].append({
                'name': name,
                'params': args,
                'qubits': qubit_indices
                })
            self._add_condition()

    def end_gate(self, name, args, qubits):
        """End a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        if name == self.in_gate:
            self.in_gate = ""
            self.listen = True
