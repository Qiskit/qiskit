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
=============================================
Circuit Converters (:mod:`qiskit.converters`)
=============================================

.. currentmodule:: qiskit.converters

.. autosummary::
   :toctree: ../stubs/

   circuit_to_dag
   dag_to_circuit
   circuit_to_instruction
   circuit_to_gate
   ast_to_dag
"""

from .circuit_to_dag import circuit_to_dag
from .dag_to_circuit import dag_to_circuit
from .circuit_to_instruction import circuit_to_instruction
from .circuit_to_gate import circuit_to_gate
from .ast_to_dag import ast_to_dag


def isinstanceint(obj):
    """ Like isinstance(obj,int), but with casting. Except for strings."""
    if isinstance(obj, str):
        return False
    try:
        int(obj)
        return True
    except TypeError:
        return False


def isinstancelist(obj):
    """ Like isinstance(obj, list), but with casting. Except for strings and dicts."""
    if isinstance(obj, (str, dict)):
        return False
    try:
        list(obj)
        return True
    except TypeError:
        return False
