# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

import random
import uuid

import numpy

from qiskit import QuantumProgram


class RandomQasmGenerator(object):
    """
    Generate random size circuits for profiling.
    """
    def __init__(self, seed=None,
                 max_qubits=5, min_qubits=1,
                 max_depth=100, min_depth=1):
        """
        Args:
            seed (int): Random number seed. If none, don't seed the generator.
            max_qubits (int): Maximum number of qubits in a circuit.
            min_qubits (int): Minimum number of qubits in a circuit.
            max_depth (int): Maximum number of operations in a circuit.
            min_depth (int): Minimum number of operations in a circuit.
        """
        self.max_depth = max_depth
        self.max_qubits = max_qubits
        self.min_depth = min_depth
        self.min_qubits = min_qubits
        self.qprogram = QuantumProgram()
        self.qreg = self.qprogram.create_quantum_register('qr', max_qubits)
        self.creg = self.qprogram.create_classical_register('cr', max_qubits)
        self.circuit_name_list = []
        self.n_qubit_list = []
        self.depth_list = []
        if seed is not None:
            random.seed(a=seed)

    def add_circuits(self, n_circuits, do_measure=True):
        """Adds circuits to program.

        Generates a circuit with a random number of operations equally weighted
        between U3 and CX. Also adds a random number of measurements in
        [1,nQubits] to end of circuit.

        Args:
          n_circuits (int): Number of circuits to add.
          do_measure (boolean): whether to add measurements
        """
        self.circuit_name_list = []
        self.n_qubit_list = numpy.random.choice(
            range(self.min_qubits, self.max_qubits + 1), size=n_circuits)
        self.depth_list = numpy.random.choice(
            range(self.min_depth, self.max_depth + 1), size=n_circuits)
        for i in range(n_circuits):
            circuit_name = str(uuid.uuid4())
            self.circuit_name_list.append(circuit_name)
            n_qubits = self.n_qubit_list[i]
            depth = self.depth_list[i]
            circuit = self.qprogram.create_circuit(circuit_name,
                                                   [self.qreg], [self.creg])
            for _ in range(depth):
                if n_qubits == 1:
                    op_ind = 0
                else:
                    op_ind = random.randint(0, 1)
                if op_ind == 0:  # U3
                    qind = random.randint(0, n_qubits-1)
                    circuit.u3(random.random(), random.random(), random.random(),
                               self.qreg[qind])
                elif op_ind == 1:  # CX
                    source, target = random.sample(range(n_qubits), 2)
                    circuit.cx(self.qreg[source], self.qreg[target])
            n_measure = random.randint(1, n_qubits)
            for _ in range(n_measure):
                qind = random.randint(0, n_qubits-1)
                if do_measure:
                    # doing this if here keeps the RNG from depending on
                    # whether measurements are done.
                    circuit.measure(self.qreg[qind], self.creg[qind])

    def get_circuit_names(self):
        return self.circuit_name_list

    def get_program(self):
        return self.qprogram
