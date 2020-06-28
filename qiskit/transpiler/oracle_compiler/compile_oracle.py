# -*- coding: utf-8 -*-

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

"""=============================================
Oracle compiler (:mod:`qiskit.transpiler.oracle_compiler.compile_oracle`)
=============================================

.. currentmodule:: qiskit.transpiler.oracle_compiler.compile_oracle

.. autofunction:: execute
"""

import inspect
from .oracle import Oracle


def compile_oracle(func):
    """
    Parses and type checks the callable ``func`` to compile it into an ``Oracle``
    that can be synthesised into a ``QuantumCircuit``.

    Args:
        func (callable): A callable (with type hints) to compile into a logic network.

    Returns:
        Oracle: An object that can synthesis into a QuantumCircuit (via ``synth()`` method).
    """
    source = inspect.getsource(func).strip()
    return Oracle(source)
