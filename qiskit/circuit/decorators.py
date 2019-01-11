"""This module contains decorators for expanding registers objects or
list of qubits into a series of single qubit/cbit instructions to be handled by the """

from functools import wraps
from qiskit.exceptions import QiskitError
from .instructionset import InstructionSet
from .quantumregister import QuantumRegister


def _1q_gate(func):
    """Wrapper for one qubit gate"""
    @wraps(func)
    def wrapper(self, *args):
        """Wrapper for one qubit gate"""
        params = args[0:-1] if len(args) > 1 else tuple()
        q = args[-1]
        if isinstance(q, QuantumRegister):
            q = [(q, j) for j in range(len(q))]

        if q and isinstance(q, list):
            instructions = InstructionSet()
            for qubit in q:
                self._check_qubit(qubit)
                instructions.add(func(self, *params, qubit))
            return instructions
        return func(self, *params, q)
    return wrapper


def _2q_gate(func):
    """Expand register or slice in two qubit gate where the number of qubits in
    each argument must match"""
    @wraps(func)
    def wrapper(self, *args):
        """Wrapper for one qubit gate"""
        params = args[0:-2] if len(args) > 2 else tuple()
        qubit1 = args[-2]
        qubit2 = args[-1]
        if isinstance(qubit1, QuantumRegister):
            qubit1 = [(qubit1, j) for j in range(len(qubit1))]
        if isinstance(qubit2, QuantumRegister):
            qubit2 = [(qubit2, j) for j in range(len(qubit2))]

        if isinstance(qubit1, list) and isinstance(qubit2, list):
            if len(qubit1) != len(qubit2):
                raise QiskitError('lengths of qubit arguments do not match: '
                                  '{0} != {1}'.format(len(qubit1), len(qubit2)))

        if qubit1 and qubit2 and isinstance(qubit1, list) and isinstance(qubit2, list):
            instructions = InstructionSet()
            for iqubit1, iqubit2 in zip(qubit1, qubit2):
                self._check_qubit(iqubit1)
                self._check_qubit(iqubit2)
                instructions.add(func(self, *params, iqubit1, iqubit2))
            return instructions
        return func(self, *params, qubit1, qubit2)
    return wrapper


def _control_target_gate(func):
    """Wrapper for two qubit control-target type gate.

    This wrapper allows the length of the target to be 1 or the same length as the control. """

    @wraps(func)
    def wrapper(self, *args):
        """Wrapper for control-target gate"""
        params = args[0:-2] if len(args) > 2 else tuple()
        ctl = args[-2]
        tgt = args[-1]
        if isinstance(ctl, QuantumRegister):
            ctl = [(ctl, i) for i in range(len(ctl))]
        if isinstance(tgt, QuantumRegister):
            tgt = [(tgt, i) for i in range(len(tgt))]
        if isinstance(ctl, list) != isinstance(tgt, list):
            # TODO: check for Qubit instance
            if isinstance(ctl, tuple):
                ctl = [ctl]
            elif isinstance(tgt, tuple):
                tgt = [tgt]
            else:
                raise QiskitError('control or target are not qubits')

        if ctl and tgt and isinstance(ctl, list) and isinstance(tgt, list):
            instructions = InstructionSet()
            if len(ctl) == len(tgt):
                for ictl, itgt in zip(ctl, tgt):
                    instructions.add(func(self, *params, ictl, itgt))
            elif len(ctl) == 1:
                for itgt in tgt:
                    instructions.add(func(self, *params, ctl[0], itgt))
            elif len(tgt) == 1:
                for ictl in ctl:
                    instructions.add(func(self, *params, ictl, tgt[0]))
            else:
                raise QiskitError('indeterminate control or target qubits')
            return instructions
        return func(self, *params, ctl, tgt)
    return wrapper
