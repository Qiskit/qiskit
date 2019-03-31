# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

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
           'TNC']

try:
    import nlopt
    import logging
    logger = logging.getLogger(__name__)
    logger.info('NLopt version: {}.{}.{}'.format(nlopt.version_major(), nlopt.version_minor(), nlopt.version_bugfix()))
    from .nlopts.crs import CRS
    from .nlopts.direct_l import DIRECT_L
    from .nlopts.direct_l_rand import DIRECT_L_RAND
    from .nlopts.esch import ESCH
    from .nlopts.isres import ISRES
    __all__ += ['CRS', 'DIRECT_L', 'DIRECT_L_RAND', 'ESCH', 'ISRES']
except ImportError:
    pass
