# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unitary gate."""
import copy
from qiskit.exceptions import QiskitError
from .instruction import Instruction


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, num_qubits, params, label=None):
        """Create a new gate.

        Args:
            name (str): the Qobj name of the gate
            num_qubits (int): the number of qubits the gate acts on.
            params (list): a list of parameters.
            label (str or None): An optional label for the gate [Default: None]
        """
        self._label = label
        super().__init__(name, num_qubits, 0, params)

    def to_matrix(self):
        """Return a Numpy.array for the gate unitary matrix.

        Raises:
            QiskitError: If a Gate subclass does not implement this method an
                exception will be raised when this base class method is called.
        """
        raise QiskitError("to_matrix not defined for this {}".format(type(self)))

    def _return_repeat(self, exponent):
        return Gate(name="%s*%s" % (self.name, exponent), num_qubits=self.num_qubits,
                    params=self.params)

    def assemble(self):
        """Assemble a QasmQobjInstruction"""
        instruction = super().assemble()
        if self.label:
            instruction.label = self.label
        return instruction

    @property
    def label(self):
        """Return gate label"""
        return self._label

    @label.setter
    def label(self, name):
        """Set gate label to name

        Args:
            name (str or None): label to assign unitary

        Raises:
            TypeError: name is not string or None.
        """
        if isinstance(name, (str, type(None))):
            self._label = name
        else:
            raise TypeError('label expects a string or None')

    def q_if(self, num_ctrl_qubits=1, label=None):
        """Return controlled version of gate
        
        Args:
            num_ctrl_qubits (int): number of controls to add to gate (default=1)
            label (str): optional gate label
        Returns:
            ControlledGate: controlled version of gate. This default algorithm
                uses num_ctrl_qubits-1 ancillae qubits so returns a gate of size
                num_qubits + 2*num_ctrl_qubits - 1.
        """
        basis_gates = {'u1', 'u3', 'id', 'cx'}
        import qiskit.circuit.controlledgate as controlledgate
        from qiskit.circuit import QuantumRegister
        import qiskit.extensions.standard.ccx as ccx
        new_num_qubits = num_ctrl_qubits + self.num_qubits
        if isinstance(self, controlledgate.ControlledGate):
            new_num_ctrl_qubits = self.num_ctrl_qubits + num_ctrl_qubits
        else:
            new_num_ctrl_qubits = num_ctrl_qubits
        print('q_if')
        if (hasattr(self, 'definition')
            and self.definition is not None
            and self.name not in basis_gates):
            definition = []
            print('definition')
            print(len(self.definition), num_ctrl_qubits)
            if num_ctrl_qubits > 0:
                bgate = self._unroll_gate(basis_gates=basis_gates)
                qr = QuantumRegister(self.num_qubits + 1)
                for rule in bgate.definition:
                    if isinstance(rule[0], Gate):
                        bgate_bits = list(
                            [qr[0]]
                            + [qr[1 + bit.index] for bit in rule[1]])
                        q_if_rule = (rule[0].q_if(), bgate_bits, [])
                        definition.append(q_if_rule)
                    else:
                        raise QiskitError('gate contains non-controllable intructions')
                if isinstance(self, controlledgate.ControlledGate):
                    new_num_ctrl_qubits = self.num_ctrl_qubits + 1
                else:
                    new_num_ctrl_qubits = 1
                cgate = controlledgate.ControlledGate('c{0:d}{1}'.format(
                    new_num_ctrl_qubits, self.name),
                    self.num_qubits+1,
                    self.params,
                    label=label,
                    num_ctrl_qubits =new_num_ctrl_qubits,
                    definition=definition)
                return cgate.q_if(num_ctrl_qubits - 1)
            else:
                cgate = controlledgate.ControlledGate('c{0:d}{1}'.format(
                    num_ctrl_qubits, self.name),
                    new_num_qubits,
                    self.params,
                    label=label,
                    num_ctrl_qubits=new_num_ctrl_qubits)
                    #definition=copy.copy(self.definition))
        else:
            print('no definition')
            definiiton = []
            if num_ctrl_qubits > 0:
                qr = QuantumRegister(self.num_qubits + 1)
                definition = [(self, qr, [])]
                if isinstance(self, controlledgate.ControlledGate):
                    new_num_ctrl_qubits = self.num_ctrl_qubits + 1
                else:
                    new_num_ctrl_qubits = 1
                return self.q_if(num_ctrl_qubits - 1)
            else:
                import ipdb;ipdb.set_trace()
                cgate = controlledgate.ControlledGate('c{0:d}{1}'.format(
                    num_ctrl_qubits, self.name),
                    new_num_qubits,
                    self.params,
                    label=label,
                    num_ctrl_qubits=new_num_ctrl_qubits)
        return cgate

    def _gate_to_circuit(self):
        from qiskit.circuit import QuantumCircuit, QuantumRegister
        qr = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(qr, name=self.name)
        for rule in self.definition:
            qc.append(rule[0], qargs=[qr[bit.index] for bit in rule[1]], cargs=[])
        return qc
        
    def _unroll_gate(self, basis_gates=['u1', 'u2', 'u3', 'id', 'cx']):
        from qiskit.converters import circuit_to_dag, dag_to_circuit, instruction_to_gate
        from qiskit.transpiler.passes import Unroller
        unroller = Unroller(basis_gates)
        dag = circuit_to_dag(self._gate_to_circuit())
        qc = dag_to_circuit(unroller.run(dag))
        return instruction_to_gate(qc.to_instruction())

    @staticmethod
    def _broadcast_single_argument(qarg):
        """Expands a single argument.

        For example: [q[0], q[1]] -> [q[0]], [q[1]]
        """
        # [q[0], q[1]] -> [q[0]]
        #              -> [q[1]]
        for arg0 in qarg:
            yield [arg0], []

    @staticmethod
    def _broadcast_2_arguments(qarg0, qarg1):
        if len(qarg0) == len(qarg1):
            # [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]]
            #                              -> [q[1], r[1]]
            for arg0, arg1 in zip(qarg0, qarg1):
                yield [arg0, arg1], []
        elif len(qarg0) == 1:
            # [[q[0]], [r[0], r[1]]] -> [q[0], r[0]]
            #                        -> [q[0], r[1]]
            for arg1 in qarg1:
                yield [qarg0[0], arg1], []
        elif len(qarg1) == 1:
            # [[q[0], q[1]], [r[0]]] -> [q[0], r[0]]
            #                        -> [q[1], r[0]]
            for arg0 in qarg0:
                yield [arg0, qarg1[0]], []
        else:
            raise QiskitError('Not sure how to combine these two qubit arguments:\n %s\n %s' %
                              (qarg0, qarg1))

    @staticmethod
    def _broadcast_3_or_more_args(qargs):
        if all(len(qarg) == len(qargs[0]) for qarg in qargs):
            for arg in zip(*qargs):
                yield list(arg), []
        else:
            raise QiskitError(
                'Not sure how to combine these qubit arguments:\n %s\n' % qargs)

    def broadcast_arguments(self, qargs, cargs):
        """Validation and handling of the arguments and its relationship.

        For example:
        `cx([q[0],q[1]], q[2])` means `cx(q[0], q[2]); cx(q[1], q[2])`. This method
        yields the arguments in the right grouping. In the given example::

            in: [[q[0],q[1]], q[2]],[]
            outs: [q[0], q[2]], []
                  [q[1], q[2]], []

        The general broadcasting rules are:
         * If len(qargs) == 1::

                [q[0], q[1]] -> [q[0]],[q[1]]

         * If len(qargs) == 2::

                [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]], [q[1], r[1]]
                [[q[0]], [r[0], r[1]]]       -> [q[0], r[0]], [q[0], r[1]]
                [[q[0], q[1]], [r[0]]]       -> [q[0], r[0]], [q[1], r[0]]

         * If len(qargs) >= 3::

                [q[0], q[1]], [r[0], r[1]],  ...] -> [q[0], r[0], ...], [q[1], r[1], ...]

        Args:
            qargs (List): List of quantum bit arguments.
            cargs (List): List of classical bit arguments.

        Returns:
            Tuple(List, List): A tuple with single arguments.

        Raises:
            QiskitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """
        if len(qargs) != self.num_qubits or cargs:
            import ipdb;ipdb.set_trace()
            raise QiskitError(
                'The amount of qubit/clbit arguments does not match the gate expectation.')

        if any([not qarg for qarg in qargs]):
            raise QiskitError('One or more of the arguments are empty')

        if len(qargs) == 1:
            return Gate._broadcast_single_argument(qargs[0])
        elif len(qargs) == 2:
            return Gate._broadcast_2_arguments(qargs[0], qargs[1])
        elif len(qargs) >= 3:
            return Gate._broadcast_3_or_more_args(qargs)
        else:
            raise QiskitError('This gate cannot handle %i arguments' % len(qargs))
