# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum information utility functions for operators.
"""

from qiskit.quantum_info.operators.linear_op import LinearOp


def commutator(a: LinearOp, b: LinearOp) -> LinearOp:
    r"""Compute commutator of a and b.

    .. math::

        ab - ba.

    Args:
        a: Operator a.
        b: Operator b.
    Returns:
        The commutator
    """
    return a @ b - b @ a


def anti_commutator(a: LinearOp, b: LinearOp) -> LinearOp:
    r"""Compute anti-commutator of a and b.

    .. math::

        ab + ba.

    Args:
        a: Operator a.
        b: Operator b.
    Returns:
        The anti-commutator
    """
    return a @ b + b @ a


def double_commutator(a: LinearOp, b: LinearOp, c: LinearOp, *, commutes: bool = True) -> LinearOp:
    r"""Compute symmetric double commutator of a, b and c.
    See also Equation (13.6.18) in [1].

    If `commutes` is `True`, it returns

    .. math::

         [[A, B], C]/2 + [A, [B, C]]/2
         = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

    If `commutes` is `False`, it returns

    .. math::
         \lbrace[A, B], C\rbrace/2 + \lbrace A, [B, C]\rbrace/2
         = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.

    Args:
        a: Operator a.
        b: Operator b.
        c: Operator c.
        commutes: True commutes, False anti-commutes.

    Returns:
        The double commutator

    References:
        [1]: R. McWeeny.
            Methods of Molecular Quantum Mechanics.
            2nd Edition, Academic Press, 1992.
            ISBN 0-12-486552-6.
    """
    sign_num = -1 if commutes else 1

    ab = a @ b
    ba = b @ a
    ac = a @ c
    ca = c @ a

    abc = ab @ c
    cba = c @ ba
    bac = ba @ c
    cab = c @ ab
    acb = ac @ b
    bca = b @ ca

    res = abc - sign_num * cba + 0.5 * (-bac + sign_num * cab - acb + sign_num * bca)

    return res
