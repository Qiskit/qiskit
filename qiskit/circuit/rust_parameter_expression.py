# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
ParameterExpression Class to enable creating simple expressions of Parameters.
"""

from __future__ import annotations
import qiskit._accelerate.circuit

SymbolExpr = qiskit._accelerate.circuit.ParameterExpression
Symbol = qiskit._accelerate.circuit.Parameter


# class Parameter(SymbolExpr):
#     def __new__(self, name: str):
#         """
#         Args:
#             name: The name of the parameter symbol.
#         """
#         symbol = Symbol(name)

#         instance = SymbolExpr.from_parameter(symbol)
#         instance._symbol = Symbol(name)
#         return instance

#         # SymbolExpr.__init__(self, name)

#     def uuid(self):
#         return self._symbol.uuid()
