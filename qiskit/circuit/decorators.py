# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
This module contains decorators for expanding register objects or
list of qubits into a series of single qubit/cbit instructions to be handled by
the wrapped operation.
"""

import functools
from qiskit.exceptions import QiskitError
from .instructionset import InstructionSet
from .register import Register
from .quantumregister import QuantumRegister
from .classicalregister import ClassicalRegister


def _is_bit(obj):
    """Determine if obj is a bit"""
    # If there is a bit type this could be replaced by isinstance.
    if isinstance(obj, tuple) and len(obj) == 2:
        if isinstance(obj[0], Register) and isinstance(obj[1], int) and obj[1] < len(obj[0]):
            return True
    return False

def _op_expand(n_qbits, func=None, n_cbits=0, broadcastable=None):
    """Decorator for expanding an operation across a whole register or register subset.
    Args:
        n_qbits (int): the number of register qubit arguments the decorated function takes
        func (function): used for decorators with keyword args
        n_cbits (int): the number of register clbit arguments the decorated function takes
        broadcastable (list(bool)): list of bool for which register args can be
            broadcast from 1 bit to the max size of the rest of the args. Defaults
            to all True if not specified.

    Return:
        type: partial function object
    """
    if func is None:
        return functools.partial(_op_expand, n_qbits, n_cbits=n_cbits, broadcastable=broadcastable)

    @functools.wraps(func)
    def wrapper(self, *args):
        n_bits = n_qbits + n_cbits
        params = args[0:-n_bits] if len(args) > n_bits else tuple()
        rargs = list(args[-n_bits:])

        # Convert items to [qu|cl]bits
        if any([isinstance(item, int) for item in rargs]):
            rqargs = rargs[:n_qbits]
            rcargs = [] if n_cbits ==0 else rargs[-n_cbits:]
            flat_qbit_list = [qbit for qreg in self.qregs for qbit in qreg]
            flat_cbit_list = [cbit for creg in self.cregs for cbit in creg]
            try:
                for index, qarg in enumerate(rqargs):
                    rqargs[index] = flat_qbit_list[qarg]
                for index, carg in enumerate(rcargs):
                    rcargs[index] = flat_cbit_list[carg]
            except IndexError:
                raise QiskitError("The integer param is out of range")
            rargs = rqargs+rcargs

        if broadcastable is None:
            blist = [True] * len(rargs)
        else:
            blist = broadcastable

        if not all([ _is_bit(arg) for arg in rargs]):
            rarg_size = [1] * n_bits
            for iarg, arg in enumerate(rargs):
                if isinstance(arg, Register):
                    rarg_size[iarg] = len(arg)
                elif isinstance(arg, list) and all([_is_bit(bit) for bit in arg]):
                    rarg_size[iarg] = len(arg)
                elif _is_bit(arg):
                    rarg_size[iarg] = 1
                else:
                    raise QiskitError('operation arguments must be qubits/cbits')
            broadcast_size = max(rarg_size)
            expanded_rargs = []
            for arg, broadcast in zip(rargs, blist):
                if isinstance(arg, Register):
                    arg = [(arg, i) for i in range(len(arg))]
                elif isinstance(arg, tuple):
                    arg = [arg]
                # now we should have a list of qubits
                if isinstance(arg, list) and len(arg) == 1 and broadcast:
                    arg = arg * broadcast_size
                if len(arg) != broadcast_size:
                    raise QiskitError('register size error')
                expanded_rargs.append(arg)
            rargs = expanded_rargs
            if all([isinstance(arg, list) for arg in rargs]):
                if all(rargs):
                    instructions = InstructionSet()
                    for irargs in zip(*rargs):
                        instructions.add(func(self, *params, *irargs),
                                         [i for i in irargs if isinstance(i[0], QuantumRegister)],
                                         [i for i in irargs if isinstance(i[0], ClassicalRegister)])
                    return instructions
                else:
                    raise QiskitError('empty control or target argument')
        return func(self, *params, *rargs)
    return wrapper
