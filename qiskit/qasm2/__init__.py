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

"""
===========================
Qasm2 (:mod:`qiskit.qasm2`)
===========================

.. currentmodule:: qiskit.qasm2

.. autosummary::
   :toctree: ../stubs/

   Qasm
   QasmError
   load()
   dump()

Pygments
========

.. autosummary::
   :toctree: ../stubs/

   OpenQASMLexer
   QasmHTMLStyle
   QasmTerminalStyle

"""
from numpy import pi
from .qasm import Qasm
from .exceptions import QasmError
from .grammar.qasm2Lexer import qasm2Lexer
from .grammar.qasm2Parser import qasm2Parser
from .grammar.qasm2Listener import qasm2Listener

from .qasm2astelem import *
from .qasm2ast import Qasm2AST
from .qasm2listener import Qasm2Listener

from .grammar.expressionLexer import expressionLexer
from .grammar.expressionParser import expressionParser
from .grammar.expressionListener import expressionListener

from .qasm2expression import Qasm2Expression
from .qasm2expressionlistener import Qasm2ExpressionListener

from .functions import load, dump

try:
    import pygments

    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

if HAS_PYGMENTS:
    try:
        from .pygments import OpenQASMLexer, QasmHTMLStyle, QasmTerminalStyle
    except Exception:  # pylint: disable=broad-except
        HAS_PYGMENTS = False
