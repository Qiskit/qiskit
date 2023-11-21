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

"""
=================================================
Qasm Pygments tools (:mod:`qiskit.qasm.pygments`)
=================================================

.. currentmodule:: qiskit.qasm.pygments

.. autosummary::
   :toctree: ../stubs/

   OpenQASMLexer
   QasmTerminalStyle
   QasmHTMLStyle
"""

# pylint: disable=wrong-import-position

from qiskit.utils.optionals import HAS_PYGMENTS

HAS_PYGMENTS.require_now("built-in OpenQASM 2 syntax highlighting")

from .lexer import OpenQASMLexer, QasmTerminalStyle, QasmHTMLStyle
