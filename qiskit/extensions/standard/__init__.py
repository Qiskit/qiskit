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
from .iden import IdGate
from .ms import MSGate
from .r import RGate
from .rx import RXGate, CrxGate
from .rxx import RXXGate
from .ry import RYGate, CryGate
from .rz import RZGate, CrzGate
from .rzz import RZZGate
from .s import SGate, SdgGate
from .swap import SwapGate, FredkinGate
from .t import TGate, TdgGate
from .u1 import U1Gate, Cu1Gate
from .u2 import U2Gate
from .u3 import U3Gate, Cu3Gate
from .x import XGate, CnotGate, ToffoliGate
from .y import YGate, CyGate
from .z import ZGate, CzGate
