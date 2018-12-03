# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=arguments-differ,unused-argument

"""
Backend for the unroller that creates a DAGCircuit object.
"""

import logging

from qiskit.dagcircuit import DAGCircuit
from qiskit._quantumcircuit import QuantumCircuit

from qiskit._measure import Measure
from qiskit._reset import Reset
from qiskit.extensions.standard.ubase import UBase
from qiskit.extensions.standard.cxbase import CXBase
from qiskit.extensions.standard.barrier import Barrier
from qiskit.extensions.standard.ccx import ToffoliGate
from qiskit.extensions.standard.cswap import FredkinGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.cy import CyGate
from qiskit.extensions.standard.cz import CzGate
from qiskit.extensions.standard.swap import SwapGate
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.iden import IdGate
from qiskit.extensions.standard.s import SGate
from qiskit.extensions.standard.s import SdgGate
from qiskit.extensions.standard.t import TGate
from qiskit.extensions.standard.t import TdgGate
from qiskit.extensions.standard.u0 import U0Gate
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.x import XGate
from qiskit.extensions.standard.y import YGate
from qiskit.extensions.standard.z import ZGate
from qiskit.extensions.standard.rx import RXGate
from qiskit.extensions.standard.ry import RYGate
from qiskit.extensions.standard.rz import RZGate
from qiskit.extensions.standard.cu1 import Cu1Gate
from qiskit.extensions.standard.ch import CHGate
from qiskit.extensions.standard.crz import CrzGate
from qiskit.extensions.standard.cu3 import Cu3Gate
from qiskit.extensions.standard.rzz import RZZGate

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

        # extra simulator basis that will not be unrolled to the above
        self.circuit.add_basis_element("snapshot", -1, 0, 1)
        self.circuit.add_basis_element("save", -1, 0, 1)
        self.circuit.add_basis_element("load", -1, 0, 1)
        self.circuit.add_basis_element("noise", -1, 0, 1)

        # extra user defined basis
        circuit = QuantumCircuit()  # TODO: make nicer when definitions not attached to circuit
        for b in basis:
            if b not in self.circuit.basis and b in circuit.definitions:
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

    def set_condition(self, creg_name, cval):
        """Attach a current condition.

        Args:
            creg_name (str): creg name to condition on.
            cval (int): value for the condition comparison.
        """
        creg = self.circuit.cregs[creg_name]
        self.creg = creg
        self.cval = cval

    def drop_condition(self):
        """Drop the current condition."""
        self.creg = None
        self.cval = None

    def start_gate(self, op, qargs=None, cargs=None):
        """Begin a custom gate.

        Args:
            op (Instruction): operation to apply to the dag.
            qargs (list(QuantumRegister, int)): qubits to attach to
            cargs (list(ClassicalRegister, int)): clbits to attach to

        Raises:
            BackendError: if encountering a non-basis opaque gate
        """
        if not self.listen:
            return

        if op.name not in self.circuit.basis:
            if self.circuit.gates[op.name]["opaque"]:
                raise BackendError("opaque gate %s not in basis" % op.name)
            else:
                logger.info("ignoring non-basis gate %s. Make sure the gates are "
                            "first expanded to basis via the DagUnroller.", op.name)
                return

        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        self.in_gate = op
        self.listen = False
        self.circuit.apply_operation_back(op, condition=condition)

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

    def u(self, arg, qubit):
        """Universal single qubit rotation gate.
        """
        if not self.listen:
            return
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        self.circuit.apply_operation_back(UBase(*arg, qubit), condition)

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

    def create_dag_op(self, name, args, qubits):
        """Create a DAG op node.
        """
        if name == "u0":
            op = U0Gate(args[0], qubits[0])
        elif name == "u1":
            op = U1Gate(args[0], qubits[0])
        elif name == "u2":
            op = U2Gate(args[0], args[1], qubits[0])
        elif name == "u3":
            op = U3Gate(args[0], args[1], args[2], qubits[0])
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
            op = RXGate(args[0], qubits[0])
        elif name == "ry":
            op = RYGate(args[0], qubits[0])
        elif name == "rz":
            op = RZGate(args[0], qubits[0])
        elif name == "rzz":
            op = RZZGate(args[0], qubits[0], qubits[1])
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
            op = CrzGate(args[0], qubits[0], qubits[1])
        elif name == "cu1":
            op = Cu1Gate(args[0], qubits[0], qubits[1])
        elif name == "cu3":
            op = Cu3Gate(args[0], args[1], args[2], qubits[0], qubits[1])
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
