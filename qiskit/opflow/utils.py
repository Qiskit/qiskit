# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for OperatorFlow"""

from qiskit.opflow.operator_base import OperatorBase
from qiskit.utils.deprecation import deprecate_func


@deprecate_func(
    since="0.24.0",
    package_name="qiskit-terra",
    additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
)
def commutator(op_a: OperatorBase, op_b: OperatorBase) -> OperatorBase:
    r"""
    Deprecated: Compute commutator of `op_a` and `op_b`.

    .. math::

        AB - BA.

    Args:
        op_a: Operator A
        op_b: Operator B
    Returns:
        OperatorBase: the commutator
    """
    return (op_a @ op_b - op_b @ op_a).reduce()


@deprecate_func(
    since="0.24.0",
    package_name="qiskit-terra",
    additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
)
def anti_commutator(op_a: OperatorBase, op_b: OperatorBase) -> OperatorBase:
    r"""
    Deprecated: Compute anti-commutator of `op_a` and `op_b`.

    .. math::

        AB + BA.

    Args:
        op_a: Operator A
        op_b: Operator B
    Returns:
        OperatorBase: the anti-commutator
    """
    return (op_a @ op_b + op_b @ op_a).reduce()


@deprecate_func(
    since="0.24.0",
    package_name="qiskit-terra",
    additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
)
def double_commutator(
    op_a: OperatorBase,
    op_b: OperatorBase,
    op_c: OperatorBase,
    sign: bool = False,
) -> OperatorBase:
    r"""
    Deprecated: Compute symmetric double commutator of `op_a`, `op_b` and `op_c`.
    See McWeeny chapter 13.6 Equation of motion methods (page 479)

    If `sign` is `False`, it returns

    .. math::

         [[A, B], C]/2 + [A, [B, C]]/2
         = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

    If `sign` is `True`, it returns

    .. math::
         \lbrace[A, B], C\rbrace/2 + \lbrace A, [B, C]\rbrace/2
         = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.

    Args:
        op_a: Operator A
        op_b: Operator B
        op_c: Operator C
        sign: False anti-commutes, True commutes
    Returns:
        OperatorBase: the double commutator
    """
    sign_num = 1 if sign else -1

    op_ab = op_a @ op_b
    op_ba = op_b @ op_a
    op_ac = op_a @ op_c
    op_ca = op_c @ op_a

    op_abc = op_ab @ op_c
    op_cba = op_c @ op_ba
    op_bac = op_ba @ op_c
    op_cab = op_c @ op_ab
    op_acb = op_ac @ op_b
    op_bca = op_b @ op_ca

    res = (
        op_abc
        - sign_num * op_cba
        + 0.5 * (-op_bac + sign_num * op_cab - op_acb + sign_num * op_bca)
    )

    return res.reduce()
