# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Backend for the unroller that creates a DAGCircuit object.
"""

from collections import OrderedDict

from qiskit.extensions.standard.ubase import UBase
from qiskit.extensions.standard.cxbase import CXBase
from qiskit.extensions.standard.barrier import Barrier
from qiskit._measure import Measure
from qiskit._reset import Reset
from qiskit.dagcircuit import DAGCircuit
from ._unrollerbackend import UnrollerBackend
from ._backenderror import BackendError


logger = logging.getLogger(__name__)


class DAGBackend(UnrollerBackend):
    """Backend for the unroller that creates a DAGCircuit object.

    Example::

        qasm = Qasm(filename = "teleport.qasm").parse()
        dagcircuit = Unroller(qasm, DAGBackend()).execute()
        print(dagcircuit.qasm())
    """

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        super().__init__(basis)
        self.creg = None
        self.cval = None
        self.circuit = DAGCircuit()
        if basis:
            self.basis = basis.copy()
        else:
            self.basis = []
        self.listen = True
        self.in_gate = None
        self.gates = OrderedDict()

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit."""
        self.basis = basis.copy()

    def define_gate(self, name, gatedata):
        """Record and pass down the data for this gate."""
        self.gates[name] = gatedata
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

    def u(self, arg, qubit, nested_scope=None):
        """Fundamental single qubit gate.

        arg is 3-tuple of Node expression objects.
        qubit is (reg,idx) tuple.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        print("qubit passed to dagbackend: ", qubit)
        if self.listen:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            if "U" not in self.basis:
                self.basis.append("U")
                self.circuit.add_basis_element("U", 1, 0, 3)
            theta, phi, lam = map(lambda x: x.sym(nested_scope), arg)
            self.circuit.apply_operation_back(UBase(theta, phi, lam, qubit),
                                              condition)

    def cx(self, qubit0, qubit1):
        """Fundamental two-qubit gate.

        qubit0 is (reg, idx) tuple for the control qubit.
        qubit1 is (reg, idx) tuple for the target qubit.
        """
        if self.listen:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            if "CX" not in self.basis:
                self.basis.append("CX")
                self.circuit.add_basis_element("CX", 2)
            self.circuit.apply_operation_back(CXBase(qubit0, qubit1),
                                              condition)

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (reg, idx) tuple for the input qubit.
        bit is (reg, idx) tuple for the output bit.
        """
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        if "measure" not in self.basis:
            self.basis.append("measure")
        if "measure" not in self.circuit.basis:
            self.circuit.add_basis_element("measure", 1, 1)
        self.circuit.apply_operation_back(Measure(qubit, bit), condition)

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (reg, idx) tuples.
        """
        if self.listen:
            qubits = []
            for qubitlist in qubitlists:
                for qubit in qubitlist:
                    qubits.append(qubit)
            if "barrier" not in self.basis:
                self.basis.append("barrier")
                self.circuit.add_basis_element("barrier", -1)
            self.circuit.apply_operation_back(Barrier(qubits))

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (reg, idx) tuple.
        """
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        if "reset" not in self.basis:
            self.basis.append("reset")
            self.circuit.add_basis_element("reset", 1)
        self.circuit.apply_operation_back(Reset(qubit), condition)

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

    def start_gate(self, op, nested_scope=None, extra_fields=None):
        """Begin a custom gate.

        Args:
            op (Instruction): operation to apply to the dag.
            nested_scope (list[dict]): list of dictionaries mapping expression variables
                to Node expression objects in order of increasing nesting depth.
            extra_fields: extra_fields used by non-standard instructions for now
                (e.g. snapshot)
        """
        if not self.listen:
            return

        if op.name not in self.basis:
            if self.gates[name]["opaque"]:
                raise BackendError("opaque gate %s not in basis" % op.name)
            else:
                logger.info("ignoring non-basis gate %s. Make sure the gates are
                        first expanded to basis via the DagUnroller." % op.name)
                return

        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        self.in_gate = op
        self.listen = False
        self.circuit.add_basis_element(op.name, len(op.qargs), len(op.cargs), len(op.param))
        self.circuit.apply_operation_back(op, condition)

    def end_gate(self, op, nested_scope=None):
        """End a custom gate.

        Args:
            op (Instruction): operation to apply to the dag.
            nested_scope (list[dict]): list of dictionaries mapping expression variables
                to Node expression objects in order of increasing nesting depth.
        """
        if op == self.in_gate:
            self.in_gate = None
            self.listen = True

    def get_output(self):
        """Returns the generated circuit."""
        return self.circuit
