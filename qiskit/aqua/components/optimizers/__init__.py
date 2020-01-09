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
Optimizers (:mod:`qiskit.aqua.components.optimizers`)
=====================================================
Optimizers, local and global

.. currentmodule:: qiskit.aqua.components.optimizers

Optimizer Base Class
====================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Optimizer

Local Optimizers
================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ADAM
   CG
   COBYLA
   L_BFGS_B
   NELDER_MEAD
   P_BFGS
   POWELL
   SLSQP
   SPSA
   TNC
   AQGD

Global Optimizers
=================
The global optimizers here all use NLopt for their core function and can only be
used if their dependent NLopt package is manually installed. See the following
section for installation instructions.

.. toctree::

   qiskit.aqua.components.optimizers.nlopts

The global optimizers are as follows:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CRS
   DIRECT_L
   DIRECT_L_RAND
   ESCH
   ISRES

"""

from .optimizer import Optimizer
from .adam_amsgrad import ADAM
from .cg import CG
from .cobyla import COBYLA
from .l_bfgs_b import L_BFGS_B
from .nelder_mead import NELDER_MEAD
from .p_bfgs import P_BFGS
from .powell import POWELL
from .slsqp import SLSQP
from .spsa import SPSA
from .tnc import TNC
from .aqgd import AQGD
from .nlopts.crs import CRS
from .nlopts.direct_l import DIRECT_L
from .nlopts.direct_l_rand import DIRECT_L_RAND
from .nlopts.esch import ESCH
from .nlopts.isres import ISRES

__all__ = ['Optimizer',
           'ADAM',
           'CG',
           'COBYLA',
           'L_BFGS_B',
           'NELDER_MEAD',
           'P_BFGS',
           'POWELL',
           'SLSQP',
           'SPSA',
           'TNC',
           'AQGD',
           'CRS', 'DIRECT_L', 'DIRECT_L_RAND', 'ESCH', 'ISRES']
