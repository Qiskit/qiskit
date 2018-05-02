# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
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
"""Generate random circuits."""

import random

import numpy

from qiskit import (qasm, unroll, ClassicalRegister, QuantumCircuit,
                    QuantumRegister)


def choices(population, weights=None, k=1):
    """
    Replacement for `random.choices()`, which is only available in Python 3.6+.
    TODO: drop once Python 3.6 is required by the sdk.
    """
    if weights and sum(weights) != 1:
        # Normalize the weights if needed, as numpy.random.choice requires so.
        weights = [float(i)/sum(weights) for i in weights]

    return numpy.random.choice(population, size=k, p=weights)


class RandomCircuitGenerator(object):
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
          min_qubits (int): Minimum number of operations in a cirucit.
          max_depth (int): Maximum number of operations in a circuit.
          min_depth (int): Minimum number of operations in circuit.
        """
        self.max_depth = max_depth
        self.max_qubits = max_qubits
        self.min_depth = min_depth
        self.min_qubits = min_qubits
        self.circuit_list = []
        self.n_qubit_list = []
        self.depth_list = []
        self.basis_gates = None
        self.circuit_name_list = []
        if seed is not None:
            random.seed(a=seed)
        # specify number of parameters and args for each op
        # in the standard extension. If type hints (PEP484) are followed
        # maybe we can guess this. "nregs" are the number of qubits the
        # operation uses. If nregs=0 then it means either 1 qubit or
        # 1 register. "nparams" are the number of parameters the operation takes.
        self.op_signature = {
            'barrier': {'nregs': 0, 'nparams': None},
            'ccx': {'nregs': 3, 'nparams': None},
            'ch': {'nregs': 2, 'nparams': None},
            'crz': {'nregs': 2, 'nparams': 1},
            'cswap': {'nregs': 3, 'nparams': None},
            'cu1': {'nregs': 2, 'nparams': 1},
            'cu3': {'nregs': 2, 'nparams': 3},
            'cx': {'nregs': 2, 'nparams': None},
            'cy': {'nregs': 2, 'nparams': None},
            'cz': {'nregs': 2, 'nparams': None},
            'h': {'nregs': 1, 'nparams': None},
            'iden': {'nregs': 1, 'nparams': None},
            'measure': {'nregs': 0, 'nparams': None},
            'reset': {'nregs': 1, 'nparams': None},
            'rx': {'nregs': 1, 'nparams': 1},
            'ry': {'nregs': 1, 'nparams': 1},
            'rz': {'nregs': 1, 'nparams': 1},
            's': {'nregs': 1, 'nparams': None},
            't': {'nregs': 1, 'nparams': None},
            'u1': {'nregs': 1, 'nparams': 1},
            'u2': {'nregs': 1, 'nparams': 2},
            'u3': {'nregs': 1, 'nparams': 3},
            'x': {'nregs': 1, 'nparams': None},
            'y': {'nregs': 1, 'nparams': None},
            'z': {'nregs': 1, 'nparams': None}}

    def add_circuits(self, n_circuits, do_measure=True, basis=None,
                     basis_weights=None):
        """Adds circuits to program.

        Generates a circuit with a random number of operations in `basis`.
        Also adds a random number of measurements in
        [1,nQubits] to end of circuit.

        Args:
            n_circuits (int): Number of circuits to add.
            do_measure (bool): Whether to add measurements.
            basis (list(str) or None): List of op names. If None, basis
                is randomly chosen with unique ops in (2,7)
            basis_weights (list(float) or None): List of weights
                corresponding to indices in `basis`.
        Raises:
            AttributeError: if operation is not recognized.
        """
        if basis is None:
            basis = list(random.sample(self.op_signature.keys(),
                                       random.randint(2, 7)))
            basis_weights = [1./len(basis)] * len(basis)
        if basis_weights is not None:
            assert len(basis) == len(basis_weights)
        uop_basis = basis[:]
        if basis_weights:
            uop_basis_weights = basis_weights[:]
        else:
            uop_basis_weights = None
        # remove barrier from uop basis if it is specified
        if 'barrier' in uop_basis:
            ind = uop_basis.index('barrier')
            del uop_basis[ind]
            if uop_basis_weights:
                del uop_basis_weights[ind]
        # remove measure from uop basis if it is specified
        if 'measure' in uop_basis:
            ind = uop_basis.index('measure')
            del uop_basis[ind]
            if uop_basis_weights:
                del uop_basis_weights[ind]
        # self.basis_gates = uop_basis
        self.basis_gates = basis
        self.circuit_name_list = []
        # TODO: replace choices with random.choices() when python 3.6 is
        # required.
        self.n_qubit_list = choices(
            range(self.min_qubits, self.max_qubits + 1), k=n_circuits)
        self.depth_list = choices(
            range(self.min_depth, self.max_depth + 1), k=n_circuits)
        for i_circuit in range(n_circuits):
            n_qubits = self.n_qubit_list[i_circuit]
            if self.min_regs_exceeds_nqubits(uop_basis, n_qubits):
                # no gate operation from this circuit can fit in the available
                # number of qubits.
                continue
            depth_cnt = self.depth_list[i_circuit]
            reg_pop = numpy.arange(1, n_qubits+1)
            register_weights = reg_pop[::-1].astype(float)
            register_weights /= register_weights.sum()
            max_registers = numpy.random.choice(reg_pop, p=register_weights)
            reg_weight = numpy.ones(max_registers) / float(max_registers)
            reg_sizes = rand_register_sizes(n_qubits, reg_weight)
            n_registers = len(reg_sizes)
            circuit = QuantumCircuit()
            for i_size, size in enumerate(reg_sizes):
                cr_name = 'cr' + str(i_size)
                qr_name = 'qr' + str(i_size)
                creg = ClassicalRegister(size, cr_name)
                qreg = QuantumRegister(size, qr_name)
                circuit.add(qreg, creg)
            while depth_cnt > 0:
                # TODO: replace choices with random.choices() when python 3.6
                # is required.
                op_name = choices(basis, weights=basis_weights)[0]
                if hasattr(circuit, op_name):
                    operator = getattr(circuit, op_name)
                else:
                    raise AttributeError('operation \"{0}\"'
                                         ' not recognized'.format(op_name))
                n_regs = self.op_signature[op_name]['nregs']
                n_params = self.op_signature[op_name]['nparams']
                if n_regs == 0:  # this is a barrier or measure
                    n_regs = random.randint(1, n_qubits)
                if n_qubits >= n_regs:
                    # warning: assumes op function signature specifies
                    # op parameters before qubits
                    op_args = []
                    if n_params:
                        op_args = [random.random() for _ in range(n_params)]
                    if op_name == 'measure':
                        # if measure occurs here, assume it's to do a conditional
                        # randomly select a register to measure
                        ireg = random.randint(0, n_registers-1)
                        qr_name = 'qr' + str(ireg)
                        cr_name = 'cr' + str(ireg)
                        qreg = circuit.regs[qr_name]
                        creg = circuit.regs[cr_name]
                        for qind in range(qreg.size):
                            operator(qreg[qind], creg[qind])
                        ifval = random.randint(0, (1 << qreg.size) - 1)
                        # TODO: replace choices with random.choices() when
                        # python 3.6 is required.
                        uop_name = choices(uop_basis, weights=uop_basis_weights)[0]
                        if hasattr(circuit, uop_name):
                            uop = getattr(circuit, uop_name)
                        else:
                            raise AttributeError('operation \"{0}\"'
                                                 ' not recognized'.format(uop_name))
                        unregs = self.op_signature[uop_name]['nregs']
                        unparams = self.op_signature[uop_name]['nparams']
                        if unregs == 0:  # this is a barrier or measure
                            unregs = random.randint(1, n_qubits)
                        if qreg.size >= unregs:
                            qind_list = random.sample(range(qreg.size), unregs)
                            uop_args = []
                            if unparams:
                                uop_args = [random.random() for _ in range(unparams)]
                            uop_args.extend([qreg[qind] for qind in qind_list])
                            uop(*uop_args).c_if(creg, ifval)
                        depth_cnt -= 1
                    elif op_name == 'barrier':
                        ireg = random.randint(0, n_registers-1)
                        qr_name = 'qr' + str(ireg)
                        qreg = circuit.regs[qr_name]
                        bar_args = [(qreg, mi) for mi in range(qreg.size)]
                        operator(*bar_args)
                    else:
                        # select random register
                        ireg = random.randint(0, n_registers-1)
                        qr_name = 'qr' + str(ireg)
                        qreg = circuit.regs[qr_name]
                        if qreg.size >= n_regs:
                            qind_list = random.sample(range(qreg.size), n_regs)
                            op_args.extend([qreg[qind] for qind in qind_list])
                            operator(*op_args)
                            depth_cnt -= 1
                        else:
                            break
            nmeasure = random.randint(1, n_qubits)
            m_list = random.sample(range(nmeasure), nmeasure)
            if do_measure:
                for qind in m_list:
                    rind = 0  # register index
                    cumtot = 0
                    while qind >= cumtot + circuit.regs['qr' + str(rind)].size:
                        cumtot += circuit.regs['qr' + str(rind)].size
                        rind += 1
                    qrind = int(qind - cumtot)
                    qreg = circuit.regs['qr'+str(rind)]
                    creg = circuit.regs['cr'+str(rind)]
                    circuit.measure(qreg[qrind], creg[qrind])
            self.circuit_list.append(circuit)

    def min_regs_exceeds_nqubits(self, basis, n_qubits):
        """Check whether the minimum number of qubits used by the operations
        in basis is between 1 and the number of qubits.

        Args:
            basis (list): list of basis names
            n_qubits (int): number of qubits in circuit
        Returns:
            boolean: result of the check.
        """
        return not any((n_qubits >= self.op_signature[opName]['nregs'] > 0
                        for opName in basis))

    def get_circuits(self, format_='dag'):
        """Get the compiled circuits generated.

        Args:
            format_ (str, optional): "qasm" | "qobj" | "QuantumCircuit"

        Returns:
           list: List of Compiled QuantumCircuit objects.

        Raises:
            NameError: if the output format is not valid.
        """
        if format_ == 'qasm':
            qasm_list = []
            for circuit in self.circuit_list:
                qasm_list.append(circuit.qasm())
            return qasm_list
        elif format_ == 'qobj':
            json_list = []
            for circuit in self.circuit_list:
                node_circuit = qasm.Qasm(data=circuit.qasm()).parse()
                unrolled_circuit = unroll.Unroller(
                    node_circuit,
                    unroll.JsonBackend(self.basis_gates))
                json_list.append(unrolled_circuit.execute())
            return json_list
        elif format_ == 'QuantumCircuit':
            qc_list = []
            for circuit in self.circuit_list:
                node_circuit = qasm.Qasm(data=circuit.qasm()).parse()
                unrolled_circuit = unroll.Unroller(
                    node_circuit,
                    unroll.CircuitBackend(self.basis_gates))
                qc_list.append(unrolled_circuit.execute())
            return qc_list
        # elif format is 'dag':
        #     qc_list = []
        #     for circuit in self.circuit_list:
        #         node_circuit = qasm.Qasm(data=circuit.qasm()).parse()
        #         unrolled_circuit = unroll.Unroller(
        #             node_circuit,
        #             unroll.DAGBackend(self.basis_gates))
        #         qc_list.append(unrolled_circuit.execute())
        #     return qc_list
        else:
            raise NameError('Unrecognized circuit output format: "{}"'.format(
                format_))


def rand_register_sizes(n_registers, pvals):
    """Return a randomly chosen list of nRegisters summing to nQubits."""
    vector = numpy.random.multinomial(n_registers, pvals)
    return vector[vector.nonzero()]
