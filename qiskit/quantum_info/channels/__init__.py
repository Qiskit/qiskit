# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum Channel Package

For explanation of terminology and details of operations see Ref. [1]

References:
    [1] C.J. Wood, J.D. Biamonte Smith, D.G. Cory Quant. Inf. Comp. 15, 0579-0811 (2015)
        Open access: arXiv:1111.6950 [quant-ph]
"""
from .qchannel import QChannel
from .operations import transform_channel
from .operations import conjugate_channel
from .operations import transpose_channel
from .operations import adjoint_channel
from .operations import compose
from .operations import kron
from .operations import power
from .operations import evolve_state
from .reps import SuperOp, Choi, Kraus, Stinespring, PauliTM, Chi
from . import reps


# TODO List for porting from old code:
# - measures: average_gate_fidelity, process_fidelity, diamond_norm
# - predicates: is_cptp, is_cp, is_cp, is_unitary
