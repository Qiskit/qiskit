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
   CCZGate
   HGate
   IGate
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
   XXMinusYYGate
   XXPlusYYGate
   ECRGate
   SGate
   SdgGate
   CSGate
   CSdgGate
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
from .r import RGate
from .rx import RXGate, CRXGate
from .rxx import RXXGate
from .ry import RYGate, CRYGate
from .ryy import RYYGate
from .rz import RZGate, CRZGate
from .rzz import RZZGate
from .rzx import RZXGate
from .xx_minus_yy import XXMinusYYGate
from .xx_plus_yy import XXPlusYYGate
from .ecr import ECRGate
from .s import SGate, SdgGate, CSGate, CSdgGate
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
from .z import ZGate, CZGate, CCZGate

from .multi_control_rotation_gates import mcrx, mcry, mcrz


def get_standard_gate_name_mapping():
    """Return a dictionary mapping the name of standard gates and instructions to an object for
    that name."""
    from qiskit.circuit.parameter import Parameter
    from qiskit.circuit.measure import Measure
    from qiskit.circuit.delay import Delay
    from qiskit.circuit.reset import Reset

    # Standard gates library mapping, multicontrolled gates not included since they're
    # variable width
    name_mapping = {
        "id": IGate(),
        "sx": SXGate(),
        "x": XGate(),
        "cx": CXGate(),
        "rz": RZGate(Parameter("λ")),
        "r": RGate(Parameter("ϴ"), Parameter("φ")),
        "reset": Reset(),
        "c3sx": C3SXGate(),
        "ccx": CCXGate(),
        "dcx": DCXGate(),
        "ch": CHGate(),
        "cp": CPhaseGate(Parameter("ϴ")),
        "crx": CRXGate(Parameter("ϴ")),
        "cry": CRYGate(Parameter("ϴ")),
        "crz": CRZGate(Parameter("ϴ")),
        "cswap": CSwapGate(),
        "csx": CSXGate(),
        "cu": CUGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ"), Parameter("γ")),
        "cu1": CU1Gate(Parameter("λ")),
        "cu3": CU3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
        "cy": CYGate(),
        "cz": CZGate(),
        "ccz": CCZGate(),
        "h": HGate(),
        "p": PhaseGate(Parameter("ϴ")),
        "rccx": RCCXGate(),
        "rcccx": RC3XGate(),
        "rx": RXGate(Parameter("ϴ")),
        "rxx": RXXGate(Parameter("ϴ")),
        "ry": RYGate(Parameter("ϴ")),
        "ryy": RYYGate(Parameter("ϴ")),
        "rzz": RZZGate(Parameter("ϴ")),
        "rzx": RZXGate(Parameter("ϴ")),
        "xx_minus_yy": XXMinusYYGate(Parameter("ϴ")),
        "xx_plus_yy": XXPlusYYGate(Parameter("ϴ")),
        "ecr": ECRGate(),
        "s": SGate(),
        "sdg": SdgGate(),
        "cs": CSGate(),
        "csdg": CSdgGate(),
        "swap": SwapGate(),
        "iswap": iSwapGate(),
        "sxdg": SXdgGate(),
        "t": TGate(),
        "tdg": TdgGate(),
        "u": UGate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
        "u1": U1Gate(Parameter("λ")),
        "u2": U2Gate(Parameter("φ"), Parameter("λ")),
        "u3": U3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
        "y": YGate(),
        "z": ZGate(),
        "delay": Delay(Parameter("t")),
        "measure": Measure(),
    }
    return name_mapping
