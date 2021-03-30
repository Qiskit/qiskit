# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=============================================================
Standard gates (:mod:`qiskit.circuit.library.standard_gates`)
=============================================================

.. autosummary::
   :toctree: ../stubs/

   C3XGate
   C3SXGate
   C4XGate
   CCXGate
   DCXGate
   CHGate
   CPhaseGate
   CRXGate
   CRYGate
   CRZGate
   CSwapGate
   CSXGate
   CUGate
   CU1Gate
   CU3Gate
   CXGate
   CYGate
   CZGate
   HGate
   IGate
   MSGate
   MCPhaseGate
   PhaseGate
   RCCXGate
   RC3XGate
   RXGate
   RXXGate
   RYGate
   RYYGate
   RZGate
   RZZGate
   RZXGate
   ECRGate
   SGate
   SdgGate
   SwapGate
   iSwapGate
   SXGate
   SXdgGate
   TGate
   TdgGate
   UGate
   U1Gate
   U2Gate
   U3Gate
   XGate
   YGate
   ZGate

"""

from .h import HGate, CHGate
from .i import IGate
from .p import PhaseGate, CPhaseGate, MCPhaseGate
from .ms import MSGate
from .r import RGate
from .rx import RXGate, CRXGate
from .rxx import RXXGate
from .ry import RYGate, CRYGate
from .ryy import RYYGate
from .rz import RZGate, CRZGate
from .rzz import RZZGate
from .rzx import RZXGate
from .ecr import ECRGate
from .s import SGate, SdgGate
from .swap import SwapGate, CSwapGate
from .iswap import iSwapGate
from .sx import SXGate, SXdgGate, CSXGate
from .dcx import DCXGate
from .t import TGate, TdgGate
from .u import UGate, CUGate
from .u1 import U1Gate, CU1Gate, MCU1Gate
from .u2 import U2Gate
from .u3 import U3Gate, CU3Gate
from .x import XGate, CXGate, CCXGate, C3XGate, C3SXGate, C4XGate, RCCXGate, RC3XGate
from .x import MCXGate, MCXGrayCode, MCXRecursive, MCXVChain
from .y import YGate, CYGate
from .z import ZGate, CZGate

from .multi_control_rotation_gates import mcrx, mcry, mcrz
