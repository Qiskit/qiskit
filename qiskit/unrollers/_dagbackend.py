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

from qiskit.extensions.standard.ubase import UBase
from qiskit.extensions.standard.cxbase import CXBase
from qiskit.extensions.standard.barrier import Barrier
from qiskit._measure import Measure
from qiskit._reset import Reset
from qiskit.dagcircuit import DAGCircuit
from qiskit._quantumcircuit import QuantumCircuit
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
        self.listen = True
        self.in_gate = None
        self.set_basis(basis)

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
        self.circuit.add_basis_element(op.name, len(op.qargs), len(op.cargs), len(op.param))
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
