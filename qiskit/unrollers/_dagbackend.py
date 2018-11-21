# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Backend for the unroller that creates a DAGCircuit object.
"""

import logging
from collections import OrderedDict

from qiskit._measure import Measure
from qiskit._reset import Reset
from qiskit.extensions.standard import *
from qiskit.dagcircuit import DAGCircuit
from qiskit._quantumcircuit import QuantumCircuit
from ._unrollerbackend import UnrollerBackend
from ._backenderror import BackendError


logger = logging.getLogger(__name__)


class DAGBackend(UnrollerBackend):
    """Backend for the unroller that creates a DAGCircuit object.
    """

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        super().__init__(basis)
        self.creg = None
        self.cval = None
        self.circuit = DAGCircuit()
        self.listen = True
        self.in_gate = None
        self.set_basis(basis or [])

    def set_basis(self, basis):
        """Declare the set of basis gates to emit."""

        # minimal basis
        self.circuit.add_basis_element(name="U", number_qubits=1,
                                       number_classical=0, number_parameters=3)
        self.circuit.add_basis_element("CX", 2, 0, 0)
        self.circuit.add_basis_element("measure", 1, 1, 0)
        self.circuit.add_basis_element("reset", 1, 0, 0)
        self.circuit.add_basis_element("barrier", -1)

        # extra user defined basis
        circuit = QuantumCircuit() # TODO: make nicer when definitions not attached to circuit
        for b in basis:
            if b not in self.circuit.basis:
                definition = circuit.definitions[b]
                self.circuit.add_basis_element(name=b,
                                               number_qubits=definition["n_bits"],
                                               number_classical=0,
                                               number_parameters=definition["n_args"])

    def define_gate(self, name, gatedata):
        """Record and pass down the data for this gate."""
        self.circuit.add_gate_data(name, gatedata)

    def version(self, version):
        """Accept the version string.

        v is a version number.
        """
        pass

    def new_qreg(self, qreg):
        """Create a new quantum register.

        qreg = QuantumRegister object
        """
        self.circuit.add_qreg(qreg)

    def new_creg(self, creg):
        """Create a new classical register.

        creg = ClassicalRegister object
        """
        self.circuit.add_creg(creg)

    def set_condition(self, creg, cval):
        """Attach a current condition.

        Args:
            creg (ClassicalRegister): creg to condition on.
            cval (int): value for the condition comparison.
        """
        self.creg = creg
        self.cval = cval

    def drop_condition(self):
        """Drop the current condition."""
        self.creg = None
        self.cval = None

    def start_gate(self, op, extra_fields=None):
        """Begin a custom gate.

        Args:
            op (Instruction): operation to apply to the dag.
        """
        if not self.listen:
            return

        if op.name not in self.circuit.basis:
            if self.circuit.gates[op.name]["opaque"]:
                raise BackendError("opaque gate %s not in basis" % op.name)
            else:
                logger.info("ignoring non-basis gate %s. Make sure the gates are "
                        "first expanded to basis via the DagUnroller." % op.name)
                return

        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        self.in_gate = op
        self.listen = False
        self.circuit.apply_operation_back(op, condition)

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
        return self.circuit

    def u(self, arg, qubit, nested_scope=None):
        """Universal single qubit rotation gate.
        """
        if not self.listen:
            return
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        params = map(lambda x: x.sym(nested_scope),arg)
        self.circuit.apply_operation_back(UBase(*list(params), qubit), condition)

    def cx(self, qubit0, qubit1):
        """Fundamental two-qubit gate.
        """
        if self.listen:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            self.circuit.apply_operation_back(CXBase(qubit0, qubit1), condition)

    def measure(self, qubit, bit):
        """Measurement operation.
        """
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        self.circuit.apply_operation_back(Measure(qubit, bit), condition)

    def barrier(self, qubitlists):
        """Barrier instruction.
        """
        if self.listen:
            qubits = []
            for qubit in qubitlists:
                for j, _ in enumerate(qubit):
                    qubits.append(qubit[j])
        self.circuit.apply_operation_back(Barrier(qubits))

    def reset(self, qubit):
        """Reset instruction.
        """
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        self.circuit.apply_operation_back(Reset(qubit), condition)

    def create_dag_op(self, name, args, qubits, clbits,
                      nested_scope=None, extra_fields=None):
        """Create a DAG op node.
        """
        params = [a.sym(nested_scope) for a in args]
        if name == "u0":
            op = U0Gate(params[0], qubits[0])
        elif name == "u1":
            op = U1Gate(params[0], qubits[0])
        elif name == "u2":
            op = U2Gate(params[0], params[1], qubits[0])
        elif name == "u3":
            op = U3Gate(params[0], params[1], params[2], qubits[0])
        elif name == "x":
            op = XGate(qubits[0])
        elif name == "y":
            op = YGate(qubits[0])
        elif name == "z":
            op = ZGate(qubits[0])
        elif name == "t":
            op = TGate(qubits[0])
        elif name == "tdg":
            op = TdgGate(qubits[0])
        elif name == "s":
            op = SGate(qubits[0])
        elif name == "sdg":
            op = SdgGate(qubits[0])
        elif name == "swap":
            op = SwapGate(qubits[0], qubits[1])
        elif name == "rx":
            op = RXGate(params[0], qubits[0])
        elif name == "ry":
            op = RYGate(params[0], qubits[0])
        elif name == "rz":
            op = RZGate(params[0], qubits[0])
        elif name == "rzz":
            op = RZZGate(qubits[0], qubits[1])
        elif name == "id":
            op = IdGate(qubits[0])
        elif name == "h":
            op = HGate(qubits[0])
        elif name == "cx":
            op = CnotGate(qubits[0], qubits[1])
        elif name == "cy":
            op = CyGate(qubits[0], qubits[1])
        elif name == "cz":
            op = CzGate(qubits[0], qubits[1])
        elif name == "ch":
            op = CHGate(qubits[0], qubits[1])
        elif name == "crz":
            op = CrzGate(params[0], qubits[0], qubits[1])
        elif name == "cu1":
            op = Cu1Gate(params[0], qubits[0], qubits[1])
        elif name == "cu3":
            op = Cu3Gate(params[0], params[1], params[2], qubits[0], qubits[1])
        elif name == "ccx":
            op = ToffoliGate(qubits[0], qubits[1], qubits[2])
        elif name == "cswap":
            op = FredkinGate(qubits[0], qubits[1], qubits[2])
        else:
            raise BackendError("unknown operation for name ast node name %s" % name)

        self.circuit.add_basis_element(op.name, len(op.qargs),
                                       len(op.cargs), len(op.param))
        self.start_gate(op)
        self.end_gate(op)
