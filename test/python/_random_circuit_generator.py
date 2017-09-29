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

import random

import numpy

from qiskit import (qasm, unroll, ClassicalRegister, QuantumCircuit,
                    QuantumRegister)


def choices(population, weights=None, k=1):
    """
    Replacement for `random.choices()`, which is only available in Python 3.6+.
    TODO: drop once Python 3.6 is required by the sdk.
    """
    return numpy.random.choice(population, size=k, p=weights)


class RandomCircuitGenerator():
    """
    Generate random size circuits for profiling.
    """
    def __init__(self, seed=None,
                 maxQubits=5, minQubits=1,
                 maxDepth=100, minDepth=1):
        """
        Args:
          seed (int): Random number seed. If none, don't seed the generator.
          minDepth int): Minimum number of operations in circuit.
          maxDepth (int): Maximum number of operations in a circuit.
          minQubits (int): Minimum number of operations in a cirucit.
          maxQubits (int): Maximum number of qubits in a circuit.
        """
        self.maxDepth = maxDepth
        self.maxQubits = maxQubits
        self.minDepth = minDepth
        self.minQubits = minQubits
        self.circuit_list = []
        self.nQubit_list = []
        self.depth_list = []
        self.basis_gates = None
        if seed is not None:
            random.seed(a=seed)
        # specify number of parameters and args for each op
        # in the standard extension. If type hints (PEP484) are followed
        # maybe we can guess this. "nregs" are the number of qubits the
        # operation uses. If nregs=0 then it means either 1 qubit or
        # 1 register. "nparams" are the number of parameters the operation takes.
        self.opSignature = {
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

    def add_circuits(self, nCircuits, doMeasure=True, basis=['u3', 'cx'],
                     basis_weights=None):
        """Adds circuits to program.

        Generates a circuit with a random number of operations in `basis`.
        Also adds a random number of measurements in
        [1,nQubits] to end of circuit.

        Args:
            nCircuits (int): Number of circuits to add.
            doMeasure (bool): Whether to add measurements.
            basis (list of str): List of op names.
            basis_weights (list of float or None): List of weights
                corresponding to indices in `basis`.
        """
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
        self.circuitNameList = []
        # TODO: replace choices with random.choices() when python 3.6 is
        # required.
        self.nQubit_list = choices(
            range(self.minQubits, self.maxQubits+1), k=nCircuits)
        self.depth_list = choices(
            range(self.minDepth, self.maxDepth+1), k=nCircuits)
        for iCircuit in range(nCircuits):
            nQubits = self.nQubit_list[iCircuit]
            depthCnt = self.depth_list[iCircuit]
            regpop = numpy.arange(1, nQubits+1)
            registerWeights = regpop[::-1].astype(float)
            registerWeights /= registerWeights.sum()
            maxRegisters = numpy.random.choice(regpop, p=registerWeights)
            regWeight = numpy.ones(maxRegisters) / float(maxRegisters)
            regSizes = rand_register_sizes(nQubits, regWeight)
            nRegisters = len(regSizes)
            circuit = QuantumCircuit()
            for isize, size in enumerate(regSizes):
                cr_name = 'cr' + str(isize)
                qr_name = 'qr' + str(isize)
                cr = ClassicalRegister(cr_name, size)
                qr = QuantumRegister(qr_name, size)
                circuit.add(qr, cr)
            while depthCnt > 0:
                # TODO: replace choices with random.choices() when python 3.6
                # is required.
                opName = choices(basis, weights=basis_weights)[0]
                if hasattr(circuit, opName):
                    op = getattr(circuit, opName)
                else:
                    raise AttributeError('operation \"{0}\"'
                                         ' not recognized'.format(opName))
                nregs = self.opSignature[opName]['nregs']
                nparams = self.opSignature[opName]['nparams']
                if nregs == 0:  # this is a barrier or measure
                    nregs = random.randint(1, nQubits)
                if nQubits >= nregs:
                    # warning: assumes op function signature specifies
                    # op parameters before qubits
                    op_args = []
                    if nparams:
                        op_args = [random.random() for p in range(nparams)]
                    if opName is 'measure':
                        # if measure occurs here, assume it's to do a conditional
                        # randomly select a register to measure
                        ireg = random.randint(0, nRegisters-1)
                        qr_name = 'qr' + str(ireg)
                        cr_name = 'cr' + str(ireg)
                        qreg = circuit.regs[qr_name]
                        creg = circuit.regs[cr_name]
                        for qind in range(qreg.size):
                            op(qreg[qind], creg[qind])
                        ifval = random.randint(0, (1 << qreg.size) - 1)
                        # TODO: replace choices with random.choices() when
                        # python 3.6 is required.
                        uopName = choices(uop_basis, weights=uop_basis_weights)[0]
                        if hasattr(circuit, uopName):
                            uop = getattr(circuit, uopName)
                        else:
                            raise AttributeError('operation \"{0}\"'
                                                 ' not recognized'.format(uopName))
                        unregs = self.opSignature[uopName]['nregs']
                        unparams = self.opSignature[uopName]['nparams']
                        if unregs == 0: # this is a barrier or measure
                            unregs = random.randint(1, nQubits)
                        if qreg.size >= unregs:
                            qindList = random.sample(range(qreg.size), unregs)
                            uop_args = []
                            if unparams:
                                uop_args = [random.random() for p in range(unparams)]
                            uop_args.extend([qreg[qind] for qind in qindList])
                            uop(*uop_args).c_if(creg, ifval)
                        depthCnt -= 1
                    elif opName is 'barrier':
                        ireg = random.randint(0, nRegisters-1)
                        qr_name = 'qr' + str(ireg)
                        qreg = circuit.regs[qr_name]
                        bar_args = [(qreg, mi) for mi in range(qreg.size)]
                        op(*bar_args)
                    else:
                        # select random register
                        ireg = random.randint(0, nRegisters-1)
                        qr_name = 'qr' + str(ireg)
                        try:
                            qreg = circuit.regs[qr_name]
                        except Exception as e:
                            print(e)
                            import pdb;pdb.set_trace()
                        if qreg.size >= nregs:
                            qindList = random.sample(range(qreg.size), nregs)
                            op_args.extend([qreg[qind] for qind in qindList])
                            op(*op_args)
                            depthCnt -= 1
            nmeasure = random.randint(1, nQubits)
            mList = random.sample(range(nmeasure), nmeasure)
            if doMeasure:
                for qind in mList:
                    rind = 0  # register index
                    cumtot = 0
                    while qind >= cumtot + circuit.regs['qr' + str(rind)].size:
                        cumtot += circuit.regs['qr' + str(rind)].size
                        rind += 1
                    qrind = int(qind - cumtot)
                    qreg = circuit.regs['qr'+str(rind)]
                    creg = circuit.regs['cr'+str(rind)]
                    try:
                        circuit.measure(qreg[qrind], creg[qrind])
                    except Exception as e:
                        print(e)
                        print(qrind)
                        import pdb;pdb.set_trace()
            self.circuit_list.append(circuit)

    def get_circuits(self, format='dag'):
        """Get the compiled circuits generated.

        Args:
            format (str, optional): "qasm" | "json" | "QuantumCircuit"

        Returns:
           List of Compiled QuantumCircuit objects.
        """
        if format is 'qasm':
            qasm_list = []
            for circuit in self.circuit_list:
                qasm_list.append(circuit.qasm())
            return qasm_list
        elif format is 'json':
            json_list = []
            for circuit in self.circuit_list:
                node_circuit = qasm.Qasm(data=circuit.qasm()).parse()
                unrolled_circuit = unroll.Unroller(
                    node_circuit,
                    unroll.JsonBackend(self.basis_gates))
                json_list.append(unrolled_circuit.execute().decode())
            return json_list
        elif format is 'QuantumCircuit':
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


def rand_register_sizes(nRegisters, pvals):
    """Return a randomly chosen list of nRegisters summing to nQubits
    """
    v = numpy.random.multinomial(nRegisters, pvals)
    return v[v.nonzero()]
