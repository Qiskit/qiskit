# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Variational Forms (:mod:`qiskit.aqua.components.variational_forms`)
===================================================================
Variational ansatzes

.. currentmodule:: qiskit.aqua.components.variational_forms

Variational Form Base Class
===========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VariationalForm

Variational Forms
=================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   RY
   RYRZ
   SwapRZ

"""

from .variational_form import VariationalForm
from .ry import RY
from .ryrz import RYRZ
from .swaprz import SwapRZ

__all__ = ['VariationalForm',
           'RY',
           'RYRZ',
           'SwapRZ']
