#!/usr/bin/env python

# -*- coding: utf-8 -*-
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

# This tool will print on the standard output a QASM program which instructions
# were generated randonmly. The program is valid from the point of view of
# QASM syntax but it has no valid logic. The main purpose of this tool is to 
# help in benchmarking and testing Quantum simulators.
# Usage example:
# Creates a QASM circuit file called circuit.qasm with ~100 operations 
# (CX and U3 gates), 16 Qubits, and a seed of 169
# ./random_qasm_generator.py -s 169 -d 100 -q 16 > circuit.qasm

import os
import sys
import random
import string
import argparse
import numpy

try:
    import qiskit
except ImportError as ierr:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import qiskit

from qiskit import QuantumProgram

class RandomQasmGenerator():
    """
    Generate circuits with random operations for profiling.
    """
    def __init__(self, seed=None, qubits=16, depth=40):
        """
        Args:
          seed: Random number seed. If none, don't seed the generator.
          depth: Number of operations in the circuit.
          qubits: Number of qubits in the circuit.
        """
        self.depth = depth
        self.qubits = qubits
        self.quantum_program = QuantumProgram()
        self.quantum_register = self.quantum_program.create_quantum_register(
            'qr', qubits)
        self.classical_register = self.quantum_program.create_classical_register(
            'cr', qubits)
        if seed is not None:
            random.seed(a=seed)
            
    def create_circuit(self, do_measure=True):
        """Creates a circuit

        Generates a circuit with a random number of operations equally weighted
        between U3 and CX. Also adds a random number of measurements in
        [1,self.qubits] to end of circuit.

        Args:
          do_measure (boolean): whether to add measurements

        Returns:
            A string representing the QASM circuit
        """
        circuit_name = ''.join(numpy.random.choice(
            list(string.ascii_letters + string.digits)))
        circuit = self.quantum_program.create_circuit(circuit_name, 
                                                      [self.quantum_register], 
                                                      [self.classical_register])

        for j in range(self.depth):
            if self.qubits == 1:
                op_ind = 0
            else:
                op_ind = random.randint(0, 1)
            if op_ind == 0: # U3
                qind = random.randint(0, self.qubits - 1)
                circuit.u3(random.random(), random.random(), random.random(),
                           self.quantum_register[qind])
            elif op_ind == 1: # CX
                source, target = random.sample(range(self.qubits), 2)
                circuit.cx(self.quantum_register[source], 
                           self.quantum_register[target])

        if do_measure:
            nmeasure = random.randint(1, self.qubits)            
            for j in range(nmeasure):
                qind = random.randint(0, self.qubits - 1)
                # doing this if here keeps the RNG from depending on
                # whether measurements are done.
                circuit.measure(self.quantum_register[qind], 
                                self.classical_register[qind])

        return circuit.qasm()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
                    Generates a random QASM circuit program.
                    This is useful for profiling/benchmarking.''')
    parser.add_argument('-s', '--seed', type=int, 
                    help='an integer for the seed')
    parser.add_argument('-d', '--depth', type=int,
                    help='an integer for the number of opertions present in'
                         'the circuit', default=40)
    parser.add_argument('-q', '--qubits', type=int,
                    help='an integer for the number of qubits', default=16)
    parser.add_argument('-m', '--measure', action='store_true',
                    help='Creates measurments operations at the end of the'
                         ' circuit')
    args = parser.parse_args()

    random_circuit = RandomQasmGenerator(seed=args.seed,
                                         depth=args.depth,
                                         qubits=args.qubits)
    qasm = random_circuit.create_circuit(do_measure=args.measure)
    print("%s" % qasm)
