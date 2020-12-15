# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Utility functions for OperatorFlow """

import logging
from typing import Optional, Union
import warnings

from .operator_base import OperatorBase
from . import legacy
from .legacy import LegacyBaseOperator

logger = logging.getLogger(__name__)


def commutator(
        op_a: Union[OperatorBase, LegacyBaseOperator],
        op_b: Union[OperatorBase, LegacyBaseOperator],
        op_c: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
        sign: bool = False,
) -> Union[OperatorBase, LegacyBaseOperator]:
    r"""
    Compute commutator of `op_a` and `op_b` or
    the symmetric double commutator of `op_a`, `op_b` and `op_c`.
    See McWeeny chapter 13.6 Equation of motion methods (page 479)
    | If only `op_a` and `op_b` are provided:
    |     result = A\*B - B\*A;
    |
    | If `op_a`, `op_b` and `op_c` are provided:
    |     result = 0.5 \* (2\*A\*B\*C + 2\*C\*B\*A - B\*A\*C - C\*A\*B - A\*C\*B - B\*C\*A)
    Args:
        op_a: operator a
        op_b: operator b
        op_c: operator c
        sign: False anti-commutes, True commutes
    Returns:
        OperatorBase: the commutator
    """
    # TODO: remove 3 months after 0.17.0
    if isinstance(op_a, LegacyBaseOperator):
        warnings.warn(
            "LegacyBaseOperator is deprecated as of 0.17.0, "
            "and will be removed no earlier than 3 months after that "
            "release date. You should use the `opflow` "
            "module instead.", DeprecationWarning
            )
        return legacy.commutator(op_a, op_b, op_c, sign)

    sign_num = 1 if sign else -1

    op_ab = op_a @ op_b
    op_ba = op_b @ op_a

    if op_c is None:
        res = op_ab - op_ba
    else:
        op_ac = op_a @ op_c
        op_ca = op_c @ op_a

        op_abc = op_ab @ op_c
        op_cba = op_c @ op_ba
        op_bac = op_ba @ op_c
        op_cab = op_c @ op_ab
        op_acb = op_ac @ op_b
        op_bca = op_b @ op_ca

        tmp = -op_bac + sign_num * op_cab - op_acb + sign_num * op_bca  # type: ignore
        tmp = 0.5 * tmp
        res = op_abc - op_cba * sign_num + tmp  # type: ignore

    return res.reduce()


def anti_commutator(
        op_a: OperatorBase,
        op_b: OperatorBase,
) -> OperatorBase:
    r"""
    Compute anti-commutator of `op_a` and `op_b` A\*B + B\*A;
    Args:
        op_a: operator a
        op_b: operator b
    Returns:
        OperatorBase: the anti-commutator
    """
    return (op_a @ op_b + op_b @ op_a).reduce()
