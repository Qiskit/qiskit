# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum Channel Representations Package

For explanation of terminology and details of operations see Ref. [1]

References:
    [1] C.J. Wood, J.D. Biamonte, D.G. Cory, Quant. Inf. Comp. 15, 0579-0811 (2015)
        Open access: arXiv:1111.6950 [quant-ph]
"""

from .superop import SuperOp
from .choi import Choi
from .kraus import Kraus
from .stinespring import Stinespring
from .ptm import PTM
from .chi import Chi
