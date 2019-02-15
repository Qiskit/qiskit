# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Matrix gate
"""
import numpy
import sympy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError


class UnitaryMatrixGate(Gate):
    """user specified unitary matrix gate class"""
    def __init__(self, unitary_matrix, *qargs, circuit=None):
        umatrix = numpy.array(unitary_matrix).astype(numpy.complex64)
        if not numpy.allclose(umatrix.dot(umatrix.T.conj()),
                              numpy.identity(2**len(qargs))):
            raise QiskitError('non-unitary matrix')
        super().__init__('unitary', [sympy.Matrix(umatrix)], list(qargs), circuit=circuit)
        self.matrix_rep = umatrix

    def _define_decompositions(self):
        self._decompositions = None

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.unitary(self.params[0], *self.qargs))

    def qasm(self):
        """Return a default OpenQASM string for the instruction."""
        name_param = self.name
        if self.params:
            name_param = "%s(%s)" % (name_param,
                                     repr(self.params[0].tolist()))

        name_param_arg = "%s %s;" % (name_param,
                                     ",".join(["%s[%d]" % (j[0].name, j[1])
                                               for j in self.qargs + self.cargs]))
        return self._qasmif(name_param_arg)


def unitary(self, unitary_matrix, *qargs):
    """Apply unitary matrix to specified qubits."""
    for qubit in qargs:
        self._check_qubit(qubit)
    return self._attach(UnitaryMatrixGate(unitary_matrix, *qargs, circuit=self))


QuantumCircuit.unitary = unitary
