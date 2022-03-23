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
from typing import Union

from qiskit.utils import optionals

if optionals.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


def normalized_gaussian(
    t: "Symbol",
    center: Union["Symbol", "Expr"],
    zeroed_width: Union["Symbol", "Expr"],
    sigma: Union["Symbol", "Expr"],
) -> "Expr":
    r"""Generates normalized gaussian symboric equation.

    For :math:`A=` ``amp`` and :math:`\sigma=` ``sigma``, the symbolic equation will be

    .. math::

        f(x) = A\exp\left(\left(\frac{x - \mu}{2\sigma}\right)^2 \right),

    with the center :math:`\mu=` ``duration/2``.
    Then, each output sample :math:`y` is modified according to:

    .. math::

        y \mapsto A\frac{y-y^*}{A-y^*},

    where :math:`y^*` is the value of the un-normalized Gaussian .
    This sets the endpoints to :math:`0` while preserving the amplitude at the center.
    If :math:`A=y^*`, :math:`y` is set to :math:`1`.
    The endpoints are at ``x = -1, x = duration + 1``.

    Integrated area under the full curve is ``amp * np.sqrt(2*np.pi*sigma**2)``

    Args:
        t: Symbol object represents time.
        center: Symbol or expression represents the middle point of the samples.
        zeroed_width: Symbol or expression represents the endpoints the samples.
        sigma: Symbol or expression represents Gaussian sigma.

    Returns:
        Symbolic equation.
    """
    gauss = sym.exp(-((t - center) / sigma) ** 2 / 2)

    t_edge = zeroed_width / 2
    offset = sym.exp(-(t_edge / sigma) ** 2 / 2)

    return (gauss - offset) / (1 - offset)
