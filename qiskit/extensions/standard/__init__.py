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

"""This module is deprecated, the gates moved to qiskit/circuit/library."""

import warnings
from .barrier import Barrier
from .h import HGate, CHGate
from .i import IGate
from .ms import MSGate
from .r import RGate
from .rx import RXGate, CRXGate
from .rxx import RXXGate
from .ry import RYGate, CRYGate
from .ryy import RYYGate
from .rz import RZGate, CRZGate
from .rzz import RZZGate
from .rzx import RZXGate
from .s import SGate, SdgGate
from .swap import SwapGate, CSwapGate
from .iswap import iSwapGate
from .dcx import DCXGate
from .t import TGate, TdgGate
from .u1 import U1Gate, CU1Gate, MCU1Gate
from .u2 import U2Gate
from .u3 import U3Gate, CU3Gate
from .x import (
    XGate, CXGate, CCXGate, RCCXGate, C3XGate, RC3XGate, C4XGate,
    MCXGate, MCXGrayCode, MCXRecursive, MCXVChain
)
from .y import YGate, CYGate
from .z import ZGate, CZGate

# to be converted to gates
from .boolean_logical_gates import logical_or, logical_and
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

warnings.warn('The module qiskit.extensions.standard is deprecated as of 0.14.0 and will be '
              'removed no earlier than 3 months after the release. You should import the '
              'standard gates from qiskit.circuit.library.standard_gates instead.',
              DeprecationWarning, stacklevel=2)
