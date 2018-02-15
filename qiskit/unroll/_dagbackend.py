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

"""
Backend for the unroller that creates a DAGCircuit object.
"""
from ._unrollerbackend import UnrollerBackend
from ._backenderror import BackendError
from ..dagcircuit import DAGCircuit


class DAGBackend(UnrollerBackend):
    """Backend for the unroller that creates a DAGCircuit object."""

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        super().__init__(basis)
        self.prec = 15
        self.creg = None
        self.cval = None
        self.circuit = DAGCircuit()
        if basis:
            self.basis = basis
        else:
            self.basis = []
        self.listen = True
        self.in_gate = ""
        self.gates = {}

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit."""
        self.basis = basis

    def define_gate(self, name, gatedata):
        """Record and pass down the data for this gate."""
        self.gates[name] = gatedata
        self.circuit.add_gate_data(name, gatedata)

    def version(self, version):
        """Accept the version string.

        v is a version number.
        """
        pass

    def new_qreg(self, name, size):
        """Create a new quantum register.

        name = name of the register
        sz = size of the register
        """
        self.circuit.add_qreg(name, size)

    def new_creg(self, name, size):
        """Create a new classical register.

        name = name of the register
        sz = size of the register
        """
        self.circuit.add_creg(name, size)

    def u(self, arg, qubit, nested_scope=None):
        """Fundamental single qubit gate.

        arg is 3-tuple of Node expression objects.
        qubit is (regname,idx) tuple.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if self.listen:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            if "U" not in self.basis:
                self.basis.append("U")
                self.circuit.add_basis_element("U", 1, 0, 3)
            self.circuit.apply_operation_back(
                "U", [qubit], [], list(map(lambda x: x.sym(nested_scope),
                                           arg)), condition)

    def cx(self, qubit0, qubit1):
        """Fundamental two-qubit gate.

        qubit0 is (regname, idx) tuple for the control qubit.
        qubit1 is (regname, idx) tuple for the target qubit.
        """
        if self.listen:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            if "CX" not in self.basis:
                self.basis.append("CX")
                self.circuit.add_basis_element("CX", 2)
            self.circuit.apply_operation_back("CX", [qubit0, qubit1], [],
                                              [], condition)

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        if "measure" not in self.basis:
            self.basis.append("measure")
        if "measure" not in self.circuit.basis:
            self.circuit.add_basis_element("measure", 1, 1)
        self.circuit.apply_operation_back(
            "measure", [qubit], [bit], [], condition)

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        if self.listen:
            names = []
            for qubit in qubitlists:
                for j, _ in enumerate(qubit):
                    names.append(qubit[j])
            if "barrier" not in self.basis:
                self.basis.append("barrier")
                self.circuit.add_basis_element("barrier", -1)
            self.circuit.apply_operation_back("barrier", names)

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        if "reset" not in self.basis:
            self.basis.append("reset")
            self.circuit.add_basis_element("reset", 1)
        self.circuit.apply_operation_back("reset", [qubit], [], [], condition)

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

    def start_gate(self, name, args, qubits, nested_scope=None):
        """Begin a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if self.listen and name not in self.basis \
                and self.gates[name]["opaque"]:
            raise BackendError("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            self.in_gate = name
            self.listen = False
            self.circuit.add_basis_element(name, len(qubits), 0, len(args))
            self.circuit.apply_operation_back(
                name, qubits, [], list(map(lambda x: x.sym(nested_scope),
                                           args)), condition)

    def end_gate(self, name, args, qubits, nested_scope=None):
        """End a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if name == self.in_gate:
            self.in_gate = ""
            self.listen = True

    def get_output(self):
        """Returns the generated circuit."""
        return self.circuit
