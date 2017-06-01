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

"""Backend for the unroller that composes qasm into simulator inputs.

Author: Jay Gambetta and Andrew Cross

The input is a AST and a basis set and returns a compiled simulator circuit
ready to be run in backends:

[
"local_unitary_simulator",
"local_qasm_simulator"
]

OUTPUT
compiled_circuit =
    {
    'number_of_qubits': 2,
    'number_of_cbits': 2,
    'number_of_operations': 4
    'qubit_order': {('q', 0): 0, ('v', 0): 1}
    'cbit_order': {('c', 1): 1, ('c', 0): 0},
    'qasm':
        [{
        'name': 'U',
        'theta': 1.570796326794897
        'phi': 1.570796326794897
        'lambda': 1.570796326794897
        'qubit_indices': [0],
        'gate_size': 1,
        },
        {
        'name': 'CX',
        'qubit_indices': [0, 1],
        'gate_size': 2,
        },
        {
        'name': 'reset',
        'qubit_indices': [1]
        },
        {
        'name': 'measure',
        'cbit_indices': [0],
        'qubit_indices': [0]
        }],
    }
"""
# TODO: currently only supports standard basis
# TODO: currently if gates are not supported
# TODO: think more about compiled_circuit dictionary i would like to have this
# langugage agnoistic and a complete representation of a quantum file for any
# simulator so some things to consider are remove 'number_of_operations',
# as it is just he lenght of qasm.
#
# Current thinking for conditionals is to add
#
# 'condition_type': 'equality',
# 'condition_cbits': [0,2,3],
# 'condition_value': 7,
#
# to the elements of qasm.
#

import numpy as np
from qiskit.unroll import BackendException
from qiskit.unroll import UnrollerBackend


class SimulatorBackend(UnrollerBackend):
    """Backend for the unroller that composes unitary matrices."""

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        self.circuit = {}
        self.circuit['qasm'] = []
        self._number_of_qubits = 0
        self._number_of_cbits = 0
        self._qubit_order = {}
        self._cbit_order = {}
        self._operation_order = 0
        self.prec = 15
        self.creg = None
        self.cval = None
        self.gates = {}
        self.trace = False
        if basis:
            self.basis = basis
        else:
            self.basis = []  # default, unroll to U, CX
        self.listen = True
        self.in_gate = ""
        self.printed_gates = []

    def set_trace(self, trace):
        """Set trace to True to enable."""
        self.trace = trace

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit.

        basis is a list of operation name strings.
        """
        self.basis = basis

    def _fs(self, number):
        """Format a floating point number as a string.

        Uses self.prec to determine the precision.
        """
        fmt = "{0:0.%sf}" % self.prec
        return fmt.format(number)

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
            self._qubit_order[(name, j)] = self._number_of_qubits + j
        self._number_of_qubits += size
        self.circuit['number_of_qubits'] = self._number_of_qubits
        self.circuit['qubit_order'] = self._qubit_order
        if self.trace:
            print("added %d qubits from qreg %s giving a total of %d qubits" %
                  (size, name, self._number_of_qubits))

    def new_creg(self, name, size):
        """Create a new classical register.

        name = name of the register
        sz = size of the register
        """
        assert size >= 0, "invalid creg size"

        for j in range(size):
            self._cbit_order[(name, j)] = self._number_of_cbits + j
        self._number_of_cbits += size
        self.circuit['number_of_cbits'] = self._number_of_cbits
        self.circuit['cbit_order'] = self._cbit_order
        if self.trace:
            print("added %d cbits from creg %s giving a total of %d qubits" %
                  (size, name, self._number_of_cbits))

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
            if self.trace:
                if self.creg is not None:
                    print("if(%s==%d) " % (self.creg, self.cval), end="")
                print("U(%s,%s,%s) %s[%d];" % (self._fs(arg[0]),
                                               self._fs(arg[1]),
                                               self._fs(arg[2]), qubit[0],
                                               qubit[1]))
            if self.creg is not None:
                raise BackendException("UnitarySimulator does not support if")
            qubit_indices = [self._qubit_order.get(qubit)]
            self._operation_order += 1
            self.circuit['number_of_operations'] = self._operation_order
            self.circuit['qasm'].append({
                        'gate_size': 1,
                        'name': "U",
                        'theta': arg[0],
                        'phi': arg[1],
                        'lambda': arg[2],
                        'qubit_indices': qubit_indices
                        })

    def cx(self, qubit0, qubit1):
        """Fundamental two-qubit gate.

        qubit0 is (regname, idx) tuple for the control qubit.
        qubit1 is (regname, idx) tuple for the target qubit.
        """
        if self.listen:
            if "CX" not in self.basis:
                self.basis.append("CX")
            if self.trace:
                if self.creg is not None:
                    print("if(%s==%d) " % (self.creg, self.cval), end="")
                print("CX %s[%d],%s[%d];" % (qubit0[0], qubit0[1],
                                             qubit1[0], qubit1[1]))
            if self.creg is not None:
                raise BackendException("UnitarySimulator does not support if")
            qubit_indices = [self._qubit_order.get(qubit0),
                             self._qubit_order.get(qubit1)]
            self._operation_order += 1
            self.circuit['number_of_operations'] = self._operation_order
            self.circuit['qasm'].append({
                        'gate_size': 2,
                        'name': 'CX',
                        'qubit_indices': qubit_indices,
                        })

    def measure(self, qubit, cbit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        self._operation_order += 1
        self.circuit['number_of_operations'] = self._operation_order
        qubit_indices = [self._qubit_order.get(qubit)]
        cbit_indices = [self._cbit_order.get(cbit)]
        self.circuit['qasm'].append({
                    'name': 'measure',
                    'qubit_indices': qubit_indices,
                    'cbit_indices': cbit_indices
                    })
        if self.trace:
            print("measure %s[%d] -> %s[%d];" % (qubit[0], qubit[1],
                                                 cbit[0], cbit[1]))

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        pass  # ignore barriers

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        self._operation_order += 1
        self.circuit['number_of_operations'] = self._operation_order
        qubit_indices = [self._qubit_order.get(qubit)]
        self.circuit['qasm'].append({
                    'name': 'reset',
                    'qubit_indices': qubit_indices,
                    })
        if self.trace:
            print("reset %s[%d];" % (qubit[0], qubit[1]))

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
        if self.listen and self.trace and name not in self.basis:
            print("// start %s, %s, %s" % (name, list(map(self._fs, args)),
                                           qubits))
        if self.listen and name not in self.basis \
           and self.gates[name]["opaque"]:
            raise BackendException("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            self.in_gate = name
            self.listen = False
            squbits = ["%s[%d]" % (x[0], x[1]) for x in qubits]
            if self.trace:
                if self.creg is not None:
                    print("if(%s==%d) " % (self.creg, self.cval), end="")
                print(name, end="")
                if len(args) > 0:
                    print("(%s)" % ",".join(map(self._fs, args)), end="")
                print(" %s;" % ",".join(squbits))
            if self.creg is not None:
                raise BackendException("UnitarySimulator does not support if")
            # Jay: update here for any other gates, like h, u1, u2, u3, but
            # need to decided how we handle the matrix.

    def end_gate(self, name, args, qubits):
        """End a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        if name == self.in_gate:
            self.in_gate = ""
            self.listen = True
        if self.listen and self.trace and name not in self.basis:
            print("// end %s, %s, %s" % (name, list(map(self._fs, args)),
                                         qubits))
