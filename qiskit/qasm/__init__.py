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
=========================
Qasm (:mod:`qiskit.qasm`)
=========================

.. currentmodule:: qiskit.qasm

QASM Routines
=============

.. autoclass:: Qasm


Pygments
========

.. autoclass:: OpenQASMLexer
    :class-doc-from: class

.. autoclass:: QasmHTMLStyle
    :class-doc-from: class

.. autoclass:: QasmTerminalStyle
    :class-doc-from: class
"""

from numpy import pi

from qiskit.utils.optionals import HAS_PYGMENTS

from .qasm import Qasm
from .exceptions import QasmError


def __getattr__(name):
    if name in ("OpenQASMLexer", "QasmHTMLStyle", "QasmTerminalStyle"):
        import qiskit.qasm.pygments

        return getattr(qiskit.qasm.pygments, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
