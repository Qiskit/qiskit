# -*- coding: utf-8 -*-

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

"""Standard gates."""

from .barrier import Barrier
from .h import HGate, CHGate
from .i import IGate
from .ms import MSGate
from .r import RGate
from .rccx import RCCXGate
from .rcccx import RCCCXGate
from .rx import RXGate, CRXGate
from .rxx import RXXGate
from .ry import RYGate, CRYGate
from .ryy import RYYGate
from .rz import RZGate, CRZGate
from .rzz import RZZGate
from .s import SGate, SdgGate
from .swap import SwapGate, CSwapGate
from .iswap import iSwapGate
from .dcx import DCXGate
from .t import TGate, TdgGate
from .u1 import U1Gate, CU1Gate
from .u2 import U2Gate
from .u3 import U3Gate, CU3Gate
from .x import XGate, CXGate, CCXGate
from .y import YGate, CYGate
from .z import ZGate, CZGate

# to be converted to gates
from .multi_control_u1_gate import mcu1
from .multi_control_toffoli_gate import mct
from .multi_control_rotation_gates import mcrx, mcry, mcrz

# deprecated gates, to be removed
from .i import IdGate
from .x import ToffoliGate
from .swap import FredkinGate
from .x import CnotGate
from .y import CyGate
from .z import CzGate
from .u1 import Cu1Gate
from .u3 import Cu3Gate
from .rx import CrxGate
from .ry import CryGate
from .rz import CrzGate
