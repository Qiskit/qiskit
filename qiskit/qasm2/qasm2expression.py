# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Created on Sat Jun 19 15:42:39 2021

Qasm2Expression parses and calculates the value of a mathematical expression
allowed as an Instruction invocation parameter by OpenQASM2. The expressions
were parsed out of the Qasm source by Qasm2Listener etc. and then are passed
to this class for expression parsing.

@author: jax jwoehr@softwoehr.com
"""
from typing import List, Union
import pprint
import numpy as np


class ExpReg:
    """
    Mechanical calculator
    """

    def __init__(self, op: str = None, operands: list = None, result: float = None):
        """


        Parameters
        ----------
        op : str, optional
            The default is None.
        operands : list, optional
            The default is None.
        result : float, optional
            The float result if we can calc it at ctor time.
        Returns
        -------
        None.

        """
        self.op = op
        self.operands = operands
        self.result = result
        self.index = 0

    @staticmethod
    def _resolve_number(num: Union[str, float]) -> float:
        _val = None
        if isinstance(num, float):
            _val = num
        if num == "pi" or num == "π":
            _val = np.pi
        else:
            _val = float(num)
        return _val

    def _go_resolving(self):
        while self.index < len(self.operands):
            _val = ExpReg._resolve_number(self.operands[self.index])
            while _val:
                self.operands[self.index] = _val
                self.index += 1

    def calc(self) -> float:
        """


        Returns
        -------
        float
            DESCRIPTION.

        """
        result = self.result
        if not result:
            pass
        return self.result


class ExpRegStack(List[ExpReg]):

    """Stack of parse tree nestings so we can unnest expressions"""

    def __init__(self, dbg: bool = False) -> None:
        super().__init__()
        self.dbg = dbg

    def push(self, expreg: ExpReg) -> None:
        """Push the context we're in"""
        self.append(expreg)
        if self.dbg:
            pprint.pprint(self)

    def pop(self) -> ExpReg:
        """Pop last context"""
        last = super().pop()
        if self.dbg:
            pprint.pprint(self)
        return last

    def peek(self, index: int = None) -> ExpReg:
        """
        Peek nth element of the context stack from front.
        (0,1,2 ...)
        If None peek last.
        """
        expreg = None
        if index:
            expreg = self[index]
        else:
            expreg = self[len(self) - 1]
        return expreg

    def peek_back(self, index: int = None) -> ExpReg:
        """
        Peek the nth previous element of the context stack.
        (... 2,1,0)
        If index == 0 peek last
        """
        return self[len(self) - index]


class Qasm2Expression:
    def __init__(
        self,
        input_src: str,
        param_dict: dict = None,
    ) -> None:
        """


        Parameters
        ----------
        input_src : str
            DESCRIPTION.
        param_dict: dict
            Dict of parameter names and their values. Optional.
        Returns
        -------
        None
            DESCRIPTION.

        """
        self.input_src = input_src
        self.param_dict = param_dict
        self.expregstack = ExpRegStack()
        self.result = None
