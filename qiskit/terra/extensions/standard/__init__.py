# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Standard gates."""
from .barrier import barrier
from .ccx import ccx
from .cswap import cswap
from .cx import cx
from .cxbase import cx_base
from .cy import cy
from .cz import cz
from .swap import swap
from .h import h
from .iden import iden
from .s import s, sdg
from .t import t, tdg
from .u0 import u0
from .u1 import u1
from .u2 import u2
from .u3 import u3
from .ubase import u_base
from .x import x
from .y import y
from .z import z
from .rx import rx
from .ry import ry
from .rz import rz
from .cu1 import cu1
from .ch import ch
from .crz import crz
from .cu3 import cu3
from .rzz import rzz

from .barrier import Barrier
from .ccx import ToffoliGate
from .cswap import FredkinGate
from .cx import CnotGate
from .cxbase import CXBase
from .cy import CyGate
from .cz import CzGate
from .swap import SwapGate
from .h import HGate
from .iden import IdGate
from .s import SGate
from .s import SdgGate
from .t import TGate
from .t import TdgGate
from .u0 import U0Gate
from .u1 import U1Gate
from .u2 import U2Gate
from .u3 import U3Gate
from .ubase import UBase
from .x import XGate
from .y import YGate
from .z import ZGate
from .rx import RXGate
from .ry import RYGate
from .rz import RZGate
from .cu1 import Cu1Gate
from .ch import CHGate
from .crz import CrzGate
from .cu3 import Cu3Gate
from .rzz import RZZGate
