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

.. deprecated:: 0.46.0

   The :mod:`qiskit.qasm` module has been deprecated and superseded by the :mod:`qiskit.qasm2`
   module which provides a faster more correct parser.


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

import warnings

from numpy import pi

from qiskit.utils.optionals import HAS_PYGMENTS

from .qasm import Qasm
from .exceptions import QasmError


warnings.warn(
    "The `qiskit.qasm` has been deprecated and superseded by the `qiskit.qasm2` module. "
    "`qiskit.qasm` will be removed in the Qiskit 1.0.0 release.",
    category=DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    if name in ("OpenQASMLexer", "QasmHTMLStyle", "QasmTerminalStyle"):
        import qiskit.qasm.pygments

        return getattr(qiskit.qasm.pygments, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
