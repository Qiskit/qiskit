# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring


from itertools import chain, combinations
from math import log2
from copy import deepcopy
import numpy as np

from qiskit.compiler import transpile, assemble
from qiskit.qobj import QasmQobjInstruction


class BasisState:
    """
    Should probably be moved to the Statevector or QuantumState class in Terra.
    Maintains three ways of representing a basis state:
    _state_string: qubit 0 is the LSB.
    _qubit_set: list of qubits that are '1'.
    _index: as an integer, where the ground state is 0.
    """

    def __init__(self, nqubits=None, state_string=None, index=None,
                 qubit_set=None):
        if state_string is not None:
            self._state_string = state_string
            self._nqubits = len(state_string)

            if nqubits is not None and self._nqubits != nqubits:
                raise ValueError('Mismatch number of qubits')

            self._qubit_set = [i for i, j in
                               enumerate(self._state_string[::-1])
                               if j == '1']
            if qubit_set is not None and self._qubit_set != qubit_set:
                raise ValueError('Mistmatch qubit_set')

            self._index = int(self._state_string, 2)
            if index is not None and self._index != index:
                raise ValueError('Mismatch index')
        elif qubit_set is not None:
            if nqubits is None:
                raise ValueError('Missing number of qubits')

            self._qubit_set = qubit_set
            self._nqubits = nqubits

            char_list = ['0'] * self._nqubits
            for qubit in self._qubit_set:
                char_list[self._nqubits - qubit - 1] = '1'
            self._state_string = ''.join(char_list)

            self._index = int(self._state_string, 2)
            if index is not None and self._index != index:
                raise ValueError('Mismatch index')

        elif index is not None:
            if nqubits is None:
                raise ValueError('Missing number of qubits')

            self._index = index
            self._nqubits = nqubits

            self._state_string = format(self._index,
                                        '0={}b'.format(self._nqubits))
            self._qubit_set = [i for i, j
                               in enumerate(self._state_string[::-1])
                               if j == '1']
        else:
            raise ValueError('Missing basis state information')


class EquivalenceChecker():
    def __init__(self, backend=None, backend_name=None):
        self.backend = backend
        self.backend_name = backend_name
        if self.backend is None:
            self.set_default_backend()
        self.qubit_limit = 16

    def set_default_backend(self):
        try:
            from qiskit.providers.aer import StatevectorSimulator
            self.backend_name = "C++ Statevector Simulator"
            self.backend = StatevectorSimulator()
        except ImportError:
            from qiskit.providers.basicaer import StatevectorSimulatorPy
            self.backend_name = "Python Statevector Simulator"
            self.backend = StatevectorSimulatorPy()

    def get_statevector(self, qobj):
        return self.backend.run(qobj).result().get_statevector(0)

    def reduce_statevector(self, statevector, qubit_set):
        n = int(log2(len(statevector)))
        result = [0] * (2 ** len(qubit_set))
        for k, val in enumerate(statevector):
            qubit_string = BasisState(nqubits=n, index=k)._state_string
            reduced_qubits = "".join([qubit_string[i] for i in qubit_set])
            result[BasisState(state_string=reduced_qubits)._index] += val
        return result

    def initialize_circuit(self, qobj, qubit_set):
        initialized_qobj = deepcopy(qobj)
        for qubit in qubit_set:
            # flip from |0> to |1>
            instruction = QasmQobjInstruction(name='u3',
                                              params=[np.pi, 0, np.pi],
                                              qubits=[qubit])
            initialized_qobj.experiments[0].instructions.insert(0, instruction)
        return initialized_qobj

    def compare_metrics(self, c1_spec, c2_spec, verbose=False):
        c1 = c1_spec['circuit']
        c2 = c2_spec['circuit']
        g1, g2 = c1.size(), c2.size()
        d1, d2 = c1.depth(), c2.depth()
        a1 = 0
        a2 = 0
        if c1_spec.get('inputs') is not None:
            a1 = c1.num_qubits - len(c1_spec['inputs'])
        if c2_spec.get('inputs') is not None:
            a2 = c2.num_qubits - len(c2_spec['inputs'])
        c1_metrics = (g1, d1, a1)
        c2_metrics = (g2, d2, a2)

        better_circuit = None
        if sum(c1_metrics) < sum(c2_metrics) and all([x <= y for (x, y)
                                                      in zip(c1_metrics,
                                                             c2_metrics)]):
            better_circuit = 0

        if sum(c2_metrics) < sum(c1_metrics) and all([x <= y for (x, y)
                                                      in zip(c2_metrics,
                                                             c1_metrics)]):
            better_circuit = 1

        if verbose:
            data_to_print = [["Metric", "Circuit 1", "Circuit 2"],
                             ["Gate Count", g1, g2],
                             ["Depth", d1, d2],
                             ["Ancillae", a1, a2],
                             ]
            for row in data_to_print:
                print("{:<12} {:^10} {:^10}".format(*row))
            if better_circuit is None:
                print("None is better")
            else:
                print("{} is better".format(
                    ["Circuit 1", "Circuit 2"][better_circuit]))

        return better_circuit

    def run(self, circuit1, circuit2,
            c1_inputs=None, c2_inputs=None,
            c1_outputs=None, c2_outputs=None,
            up_to_phase=False):
        if c1_inputs is not None or c2_inputs is not None \
                or c1_outputs is not None or c2_outputs is not None:
            return self.output_comparision_check(circuit1, circuit2,
                                                 c1_inputs, c2_inputs,
                                                 c1_outputs, c2_outputs,
                                                 up_to_phase)
        length1 = len(circuit1.qubits)
        length2 = len(circuit2.qubits)
        if length1 != length2:
            raise RuntimeError(
                "Qubit numbers are different ({} != {})".format(length1,
                                                                length2))
        n = len(circuit1.qubits)
        if n > self.qubit_limit:
            raise RuntimeError(
                "Cannot handle more than {} qubits".format(self.qubit_limit))
        c = circuit1 + circuit2.inverse()
        qobj = assemble(transpile(c, self.backend))
        for i in range(2 ** n):
            basis_state = BasisState(nqubits=n, index=i)
            initialized_qobj = self.initialize_circuit(qobj,
                                                       basis_state._qubit_set)
            statevec = self.get_statevector(initialized_qobj)
            s_i = statevec[i]
            if up_to_phase:
                s_i = np.abs(s_i)
            if not np.isclose(s_i, 1):
                return {"result": False,
                        "counterexample": str(basis_state._state_string),
                        "backend": self.backend_name
                        }

        return {"result": True,
                "backend": self.backend_name
                }

    def get_output_for_input(self, qobj, qubit_set, c_inputs, c_outputs):
        c_qubit_set = [c_inputs[i] for i in qubit_set]
        initialized_qobj = self.initialize_circuit(qobj, c_qubit_set)
        statevec = self.get_statevector(initialized_qobj)
        reduced_s = self.reduce_statevector(statevec, c_outputs)
        return reduced_s

    def normalize(self, statevec):
        v = max(statevec, key=np.abs)
        return [x / v for x in statevec]

    def output_comparision_check(self, circuit1, circuit2,
                                 c1_inputs, c2_inputs,
                                 c1_outputs, c2_outputs,
                                 up_to_phase=False
                                 ):
        length1 = len(c1_inputs)
        length2 = len(c2_inputs)
        if length1 != length2:
            raise RuntimeError(
                "Qubit numbers are different ({} != {})".format(length1,
                                                                length2))
        n = len(c1_inputs)
        if n > self.qubit_limit:
            raise RuntimeError(
                "Cannot handle more than {} qubits".format(self.qubit_limit))

        qobj1 = assemble(transpile(circuit1, self.backend))
        qobj2 = assemble(transpile(circuit2, self.backend))

        for qubit_set in chain.from_iterable(
                combinations(range(n), r) for r in range(n + 1)):
            s1 = self.get_output_for_input(qobj1, qubit_set, c1_inputs,
                                           c1_outputs)
            s2 = self.get_output_for_input(qobj2, qubit_set, c2_inputs,
                                           c2_outputs)

            if up_to_phase:
                s1 = self.normalize(s1)
                s2 = self.normalize(s2)

            if not np.allclose(s1, s2):
                qubit_string = BasisState(nqubits=n,
                                          qubit_set=qubit_set)._state_string
                return {"result": False,
                        "counterexample": qubit_string,
                        "backend": self.backend_name
                        }
        return {"result": True}
