# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
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


def _convert_to_bits(a_list, bits):
    """ Recursively converts the integers, tuples and ranges in a_list
    for a qu/clbit from the bits. E.g. bits[item_in_a_list]"""
    new_list = []
    for item in a_list:
        if isinstance(item, (int, slice)):
            # eg. circuit.h(2)
            # eg. circuit.h(slice(0, 2))
            try:
                new_list.append(bits[item])
            except IndexError:
                raise QiskitError("The integer param is out of range")
        elif isinstance(item, list):
            # eg. circuit.h([0, 2])
            new_list.append(_convert_to_bits(item, bits))
        elif isinstance(item, range):
            # eg. circuit.h(range(0, 2))
            new_list.append(_convert_to_bits([index for index in item], bits))
        else:
            new_list.append(item)
    return new_list


def _to_bits(nqbits, ncbits=0, func=None):
    """Convert gate arguments to [qu|cl]bits from integers, slices, ranges, etc.
    For example circuit.h(0) -> circuit.h(QuantumRegister(2)[0]) """
    if func is None:
        return functools.partial(_to_bits, nqbits, ncbits)

    @functools.wraps(func)
    def wrapper(self, *args):
        qbits = self.qubits
        cbits = self.clbits

        nparams = len(args) - nqbits - ncbits
        params = args[:nparams]
        qb_args = args[nparams:nparams + nqbits]
        cl_args = args[nparams + nqbits:]

        args = list(params) + _convert_to_bits(qb_args, qbits) + _convert_to_bits(cl_args, cbits)

        return func(self, *args)

    return wrapper


def _op_expand(n_bits, func=None, broadcastable=None):
    """Decorator for expanding an operation across a whole register or register subset.
    Args:
        n_bits (int): the number of register bit arguments the decorated function takes
        func (function): used for decorators with keyword args
        broadcastable (list(bool)): list of bool for which register args can be
            broadcast from 1 bit to the max size of the rest of the args. Defaults
            to all True if not specified.

    Return:
        type: partial function object
    """
    if func is None:
        return functools.partial(_op_expand, n_bits, broadcastable=broadcastable)

    @functools.wraps(func)
    def wrapper(self, *args):
        params = args[0:-n_bits] if len(args) > n_bits else tuple()
        rargs = args[-n_bits:]

        if broadcastable is None:
            blist = [True] * len(rargs)
        else:
            blist = broadcastable

        if not all([_is_bit(arg) for arg in rargs]):
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
