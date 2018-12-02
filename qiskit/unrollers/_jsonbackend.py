# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=arguments-differ

"""Backend for the unroller that composes qasm into json file.

The input is a AST and a basis set and returns a json memory object::

    {
      "header": {
        "n_qubits": 2, // int
        "memory_slots": 2, // int
        "qubit_labels": [["q", 0], ["q", 1], null], // list[list[string, int] or null]
        "clbit_labels": [["c", 0], ["c", 1]], // list[list[string, int]]
        "qreg_sizes": [["q", 1], ["v", 1]], // list[list[string, int]]
        "creg_sizes": [["c", 2]]  // list[list[string, int]]
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
from collections import OrderedDict
import sympy

from qiskit.unrollers._backenderror import BackendError
from qiskit.unrollers._unrollerbackend import UnrollerBackend


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
            'n_qubits': 0,
            'memory_slots': 0,
            'qubit_labels': [],
            'clbit_labels': [],
            'qreg_sizes': [],
            'creg_sizes': []
        }
        self._number_of_qubits = 0
        self._number_of_clbits = 0
        self._qreg_sizes = []
        self._creg_sizes = []
        self._qubit_order = []
        self._cbit_order = []
        self._qubit_order_internal = OrderedDict()
        self._cbit_order_internal = OrderedDict()

        self.creg = None
        self.cval = None
        self.gates = OrderedDict()
        if basis:
            self.basis = basis.copy()
        else:
            self.basis = []  # default, unroll to U, CX
        self.listen = True
        self.in_gate = ""
        self.printed_gates = []

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit.

        basis is a list of operation name strings.
        """
        self.basis = basis.copy()

    def version(self, version):
        """Print the version string.

        v is a version number.
        """
        pass

    def new_qreg(self, qreg):
        """Create a new quantum register.

        qreg = QuantumRegister object
        """
        self._qreg_sizes.append([qreg.name, qreg.size])

        # order qubits from lower to higher index. backends will do little endian.
        for j in range(qreg.size):
            self._qubit_order.append([qreg.name, j])
            self._qubit_order_internal[(qreg.name, j)] = self._number_of_qubits + j
        self._number_of_qubits += qreg.size
        # TODO: avoid rewriting the same data over and over
        self.circuit['header']['n_qubits'] = self._number_of_qubits
        self.circuit['header']['qreg_sizes'] = self._qreg_sizes
        self.circuit['header']['qubit_labels'] = self._qubit_order

    def new_creg(self, creg):
        """Create a new classical register.

        creg = ClassicalRegister object
        """
        self._creg_sizes.append([creg.name, creg.size])
        # order clbits from lower to higher index. backends will do little endian.
        for j in range(creg.size):
            self._cbit_order.append([creg.name, j])
            self._cbit_order_internal[(creg.name, j)] = self._number_of_clbits + j
        self._number_of_clbits += creg.size
        # TODO: avoid rewriting the same data over and over
        self.circuit['header']['memory_slots'] = self._number_of_clbits
        self.circuit['header']['creg_sizes'] = self._creg_sizes
        self.circuit['header']['clbit_labels'] = self._cbit_order

    def define_gate(self, name, gatedata):
        """Define a new quantum gate.

        name is a string.
        gatedata is the AST node for the gate.
        """
        self.gates[name] = gatedata

    def _add_condition(self):
        """Check for a condition (self.creg) and add fields if necessary.

        Fields are added to the last operation in the circuit.
        """
        if self.creg is not None:
            mask = 0
            for cbit, index in self._cbit_order_internal.items():
                if cbit[0] == self.creg.name:
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

    def start_gate(self, op, qargs, cargs, extra_fields=None):
        """
        Begin a custom gate.

        Args:
            op (Instruction): operation to apply to the dag.
            qargs (list[QuantumRegister, int]): qubit arguments
            cargs (list[ClassicalRegister, int]): clbit arguments
            extra_fields (dict): extra_fields used by non-standard instructions
                for now (e.g. snapshot)

        Raises:
            BackendError: if using a non-basis opaque gate
        """
        if not self.listen:
            return
        if op.name not in self.basis and self.gates[op.name]["opaque"]:
            raise BackendError("opaque gate %s not in basis" % op.name)
        if op.name in self.basis:
            self.in_gate = op
            self.listen = False
            qubit_indices = [self._qubit_order_internal.get((qubit[0].name, qubit[1]))
                             for qubit in qargs]
            clbit_indices = [self._cbit_order_internal.get((cbit[0].name, cbit[1]))
                             for cbit in cargs]
            gate_instruction = {
                'name': op.name,
                'params': list(map(lambda x: x.evalf(), op.param)),
                'texparams': list(map(sympy.latex, op.param)),
                'qubits': qubit_indices,
                'memory': clbit_indices
            }
            if extra_fields is not None:
                gate_instruction.update(extra_fields)
            self.circuit['instructions'].append(gate_instruction)
            self._add_condition()

    def end_gate(self, op):
        """End a custom gate.

        Args:
            op (Instruction): operation to apply to the dag.
        """
        if op == self.in_gate:
            self.in_gate = None
            self.listen = True

    def get_output(self):
        """Returns the generated circuit."""
        if not self._is_circuit_valid():
            raise BackendError("Invalid circuit! Please check the syntax of your circuit."
                               "Has Qasm parsing been called?. e.g: unroller.execute().")
        return self.circuit

    def _is_circuit_valid(self):
        """Checks whether the circuit object is a valid one or not."""
        return (len(self.circuit['header']) > 0 and
                len(self.circuit['instructions']) >= 0)
