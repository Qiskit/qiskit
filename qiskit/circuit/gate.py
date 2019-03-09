# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unitary gate.
"""
from .instruction import Instruction


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, num_qubits, params, circuit=None):
        """Create a new composite gate.

        name = instruction name string
        params = list of real parameters (will be converted to symbolic)
        circuit = QuantumCircuit containing this gate
        """
        self._matrix_rep = None

        super().__init__(name, num_qubits, 0, params, circuit)

    def inverse(self):
        """Invert this gate.

        If the gate is composite (i.e. has decomposition rules), then those
        rules will be recursively inverted.

        Special gates inheriting from Gate can implement their own inverse
        (e.g. T and Tdg)

        Returns:
            Gate: the inverted gate

        Raises:
            NotImplementedError: if the gate is not composite and an inverse
                has not been implemented for it.
        """
        from qiskit.circuit import QuantumCircuit
        from qiskit.converters import dag_to_circuit, circuit_to_dag
        if not self._decompositions:
            raise NotImplementedError("inverse not implemented")
        inverse_inst = self.copy(name=self.name+'_dg')
        new_decompositions = []
        for decomposition in self._decompositions:
            circ = dag_to_circuit(decomposition)
            new_circ = QuantumCircuit(*circ.qregs, *circ.cregs)
            for inst, qargs, cargs in reversed(circ.data):
                new_circ.append(inst.inverse(), qargs, cargs)
            new_decompositions.append(circuit_to_dag(new_circ))
        inverse_inst._decompositions = new_decompositions
        return inverse_inst

    def decompositions(self):
        """Returns a list of possible decompositions. """
        if self._decompositions is None:
            self._define_decompositions()
        return self._decompositions

    def _define_decompositions(self):
        """Populates self.decompositions with way to decompose this instruction."""
        raise NotImplementedError("No decomposition rules defined for %s" % self.name)

    @property
    def matrix_rep(self):
        """Return matrix representation if it exists else None"""
        return self._matrix_rep

    @matrix_rep.setter
    def matrix_rep(self, matrix):
        """Set matrix representation"""
        self._matrix_rep = matrix
