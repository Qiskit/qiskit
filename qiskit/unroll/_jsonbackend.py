# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Backend for the unroller that composes qasm into json file.

The input is a AST and a basis set and returns a json memory object::

    {
     "header": {
     "number_of_qubits": 2, // int
     "number_of_clbits": 2, // int
     "qubit_labels": [["q", 0], ["v", 0]], // list[list[string, int]]
     "clbit_labels": [["c", 2]], // list[list[string, int]]
     }
     "instructions": // list[map]
        [
            {
                "name": , // required -- string
                "params": , // optional -- list[double]
                "texparams": , // optional -- list[string]
                "qubits": , // optional -- list[int]
                "cbits": , //optional -- list[int]
                "conditional":  // optional -- map
                    {
                        "type": "equals", // string
                        "mask": "0xHexadecimalString", // big int
                        "val":  "0xHexadecimalString", // big int
                    }
            },
        ]
    }
"""
from qiskit.unroll import BackendError
from qiskit.unroll import UnrollerBackend
from qiskit import QISKitError


class JsonBackend(UnrollerBackend):
    """Backend for the unroller that makes a Json quantum circuit."""

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        The default basis is ["U", "CX"].
        """
        super().__init__(basis)
        self.circuit = {}
        self.circuit['instructions'] = []
        self.circuit['header'] = {
            'number_of_qubits': 0,
            'number_of_clbits': 0,
            'qubit_labels': [],
            'clbit_labels': []
        }
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
        self.circuit['header']['number_of_clbits'] = self._number_of_cbits
        self.circuit['header']['clbit_labels'] = self._cbit_order

    def define_gate(self, name, gatedata):
        """Define a new quantum gate.

        name is a string.
        gatedata is the AST node for the gate.
        """
        self.gates[name] = gatedata

    def u(self, arg, qubit, nested_scope=None):
        """Fundamental single-qubit gate.

        arg is 3-tuple of Node expression objects.
        qubit is (regname, idx) tuple.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if self.listen:
            if "U" not in self.basis:
                self.basis.append("U")
            qubit_indices = [self._qubit_order_internal.get(qubit)]
            self.circuit['instructions'].append({
                'name': "U",
                # TODO: keep these real for now, until a later time
                'params': [float(arg[0].real(nested_scope)),
                           float(arg[1].real(nested_scope)),
                           float(arg[2].real(nested_scope))],
                'texparams': [arg[0].latex(prec=8, nested_scope=nested_scope),
                              arg[1].latex(prec=8, nested_scope=nested_scope),
                              arg[2].latex(prec=8, nested_scope=nested_scope)],
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
                # Would be nicer to zero pad the mask, but we
                # need to know the total number of cbits.
                # format_spec = "{0:#0{%d}X}" % number_of_clbits
                # format_spec.format(mask)
                conditional = {
                    'type': "equals",
                    'mask': "0x%X" % mask,
                    'val': "0x%X" % self.cval
                }
            self.circuit['instructions'][-1]['conditional'] = conditional

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
            self.circuit['instructions'].append({
                'name': 'CX',
                'qubits': qubit_indices,
            })
            self._add_condition()

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        if "measure" not in self.basis:
            self.basis.append("measure")
        qubit_indices = [self._qubit_order_internal.get(qubit)]
        clbit_indices = [self._cbit_order_internal.get(bit)]
        self.circuit['instructions'].append({
            'name': 'measure',
            'qubits': qubit_indices,
            'clbits': clbit_indices,
            'memory': clbit_indices.copy()
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
            self.circuit['instructions'].append({
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
        self.circuit['instructions'].append({
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

    def start_gate(self, name, args, qubits, nested_scope=None, extra_fields=None):
        if self.listen and name not in self.basis \
                and self.gates[name]["opaque"]:
            raise BackendError("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            self.in_gate = name
            self.listen = False
            qubit_indices = [self._qubit_order_internal.get(qubit)
                             for qubit in qubits]
            gate_instruction = {
                'name': name,
                # TODO: keep these real for now, until a later time
                'params': list(map(lambda x: float(x.real(nested_scope)),
                                   args)),
                'texparams': list(map(lambda x:
                                      x.latex(prec=8,
                                              nested_scope=nested_scope),
                                      args)),
                'qubits': qubit_indices,
            }
            if extra_fields is not None:
                gate_instruction.update(extra_fields)
            self.circuit['instructions'].append(gate_instruction)
            self._add_condition()

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
        if not self._is_circuit_valid():
            raise QISKitError("Invalid circuit! Please check the syntax of your circuit."
                              "Has the Qasm parsing been called?. e.g: unroller.execute().")
        return self.circuit

    def _is_circuit_valid(self):
        """Checks whether the circuit object is a valid one or not."""
        return (len(self.circuit['header']) > 0 and
                len(self.circuit['instructions']) > 0)
