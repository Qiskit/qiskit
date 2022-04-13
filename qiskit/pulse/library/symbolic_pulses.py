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

# pylint: disable=invalid-name

"""Symbolic waveform module. These are pulses which are described by a
set of symbolic equations for envelope and parameter constraints.

User can define any subclass of :class:`~SymbolicPulse` that implements
arbitrary waveform. A custom pulse instance can be serialized with :mod:`~qiskit.qpy` serializer
so that one can transmit arbitrary waveform data in compact format
that only describes serializable symbolic expression along with raw parameter values.

This module can easily be extended to describe more pulse shapes. The new class should:
  - have a descriptive name
  - be a well known and/or well described formula (include the formula in the class docstring)
  - take some parameters (at least `duration`) that are defined in :attr:`SymbolicPulse.PARAM_DEF`
  - implement :meth:`SymbolicPulse._define_envelope` to describe symbolic expression for envelope
  - implement :meth:`SymbolicPulse._define_constraints` which is a list of
    symbolic expressions that describes list of validation conditions for assigned parameters.
    A symbolic pulse instance can be created when all constraints return ``True``.
    One can implement if clause as a list of three expressions for evaluating "if", "then", "else".

The defined parameter expressions are immediately lambdified and stored in the class attribute
when the symbolic pulse subclass is loaded. This cache mechanism drastically improves the
performance when waveform samples are repeatedly generated for, for example, visualization.

"""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform

# Pulse module doesn't employ symengine due to some missing features.
# In addition, there are several syntactic difference in boolean expression.
# Thanks to Lambdify at subclass instantiation, the performance regression is not significant.
# However, this will create some latency for the first sympy import.


def _normalized_gaussian(
    t: "Symbol",
    center: Union["Symbol", "Expr"],
    zeroed_width: Union["Symbol", "Expr"],
    sigma: Union["Symbol", "Expr"],
) -> "Expr":
    r"""Helper function to return normalized gaussian symboric equation.

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
    import sympy as sym

    gauss = sym.exp(-(((t - center) / sigma) ** 2) / 2)

    t_edge = zeroed_width / 2
    offset = sym.exp(-((t_edge / sigma) ** 2) / 2)

    return (gauss - offset) / (1 - offset)


def _evaluate_ite(ite_lambda_func: List[Union[Callable, List]], *values: complex) -> bool:
    """Helper function to evaluate ITE-like symbolic expression.

    Args:
        ite_lambda_func: Lambdified function which is a list of "IF", "THEN", "ELSE".
        values: Parameters to be assigned.

    Returns:
        Result of evaluation.

    Raises:
        PulseError: When ITE expression is in the wrong syntax.
    """
    try:
        if_func, then_func, else_func = ite_lambda_func
    except ValueError as ex:
        raise PulseError(
            f"Invalid ITE expression with length {len(ite_lambda_func)}. "
            "This should be a list of callables for evaluating if, then, else."
        ) from ex

    if bool(if_func(*values)):
        if isinstance(then_func, list):
            # Nested if clause
            return _evaluate_ite(then_func, *values)
        return bool(then_func(*values))
    if isinstance(else_func, list):
        # Nested if clause
        return _evaluate_ite(else_func, *values)
    return bool(else_func(*values))


def _lambdify_ite(ite_expr: List["Expr"], params: List["Symbol"]) -> List[Union[Callable, List]]:
    """Helper function to lambdify ITE-like symbolic expression.

    Args:
        ite_expr: Symbolic expression which is a list of "IF", "THEN", "ELSE".
        params: Symbols to be used in the expression.

    Returns:
        Lambdified ITE-like expression.
    """
    import sympy as sym

    lambda_lists = []
    for expr in ite_expr:
        if isinstance(expr, list):
            lambda_lists.append(_lambdify_ite(expr, params))
        else:
            lambda_lists.append(sym.lambdify(params, expr))
    return lambda_lists


class EnvelopeDescriptor:
    """Descriptor of pulse envelope.

    When the descriptor of the symbolic pulse subclass is called first time,
    it calls sympy lambdify to create callable and cache the function in the descriptor.
    This improves performance of repeated evaluation of waveform samples.
    """

    global_envelopes = {}
    source_exprs = {}

    def __get__(self, instance, owner) -> Callable:
        clsname = owner.__name__

        if clsname not in EnvelopeDescriptor.global_envelopes:
            import sympy as sym

            gendef = getattr(owner, "_define_envelope")
            source = gendef()

            params = [sym.Symbol(p) for p in ["t"] + owner.PARAM_DEF]
            EnvelopeDescriptor.global_envelopes[clsname] = sym.lambdify(params, source)
            EnvelopeDescriptor.source_exprs[clsname] = source

        return EnvelopeDescriptor.global_envelopes[clsname]


class ConstraintsDescriptor:
    """Descriptor of pulse parameter constraints.

    A symbolic pulse subclass may provide multiple expressions to validate parameters,
    thus it generates a list of callables that takes set of pulse parameters.

    When the descriptor of the symbolic pulse subclass is called first time,
    it calls sympy lambdify to create callable and cache the function in the descriptor.
    This improves performance of repeated evaluation of waveform samples.
    """

    global_constraints = {}
    source_exprs = {}

    def __get__(self, instance, owner) -> List[Union[Callable, List]]:
        clsname = owner.__name__

        if clsname not in ConstraintsDescriptor.global_constraints:
            import sympy as sym

            gendef = getattr(owner, "_define_constraints")
            source = gendef()
            if not source:
                constraints = []
            else:
                params = [sym.Symbol(p) for p in ["limit"] + owner.PARAM_DEF]
                constraints = []
                for constraint_expr in source:
                    if isinstance(constraint_expr, list):
                        # If clause
                        constraints.append(_lambdify_ite(constraint_expr, params))
                    else:
                        constraints.append(sym.lambdify(params, constraint_expr))
            ConstraintsDescriptor.global_constraints[clsname] = constraints
            ConstraintsDescriptor.source_exprs[clsname] = source

        return ConstraintsDescriptor.global_constraints[clsname]


class SymbolicPulse(Pulse):
    """The abstract superclass for parametric pulses.

    Attributes:
        PARAM_DEF (List[str]): A list of parameter names that constitutes a symbolic pulse.
            Note that "t" and "limit" are reserved by the symbolic pulse superclass.
            Subclass must define at least "duration" to define the length of instruction.
        envelope (Callable): Lambdified symbolic equation to define the waveform envelope.
            This attribute is populated when the subclass is instantiated first time.
        constraints (List[Union[Callable, List]]): Lambdified symbolic equations to
            provide validation for pulse parameters when assigned.
            This attribute is populated when the subclass is instantiated first time.
    """

    __slots__ = ("param_values",)

    PARAM_DEF = ["duration"]
    envelope = EnvelopeDescriptor()
    constraints = ConstraintsDescriptor()

    @abstractmethod
    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        parameters: Optional[Tuple[Union[ParameterExpression, complex], ...]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create a parametric pulse and validate the input parameters.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            parameters: Other parameters to form waveform.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Raises:
            PulseError: When not all parameters are listed in the attribute :attr:`PARAM_DEF`.
        """
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

        if parameters:
            self.param_values = (duration,) + tuple(parameters)
        else:
            self.param_values = (duration,)

        if len(self.param_values) != len(self.PARAM_DEF):
            raise PulseError(
                f"Number of parameters {len(self.param_values)} does not match "
                f"with these defined parameters {self.PARAM_DEF}"
            )

        # Validate parameters when they are all assigned.
        if not self.is_parameterized():
            self.validate_parameters()

    def __getattr__(self, item):
        # For backward compatibility, return parameter names as property-like

        if item not in self.PARAM_DEF:
            raise AttributeError(f"'{self.__class__.__name__}' object has not attribute '{item}'")
        return self.parameters[item]

    @classmethod
    def _define_envelope(cls) -> "Expr":
        """Return symbolic expression of pulse waveform.

        A custom pulse without having this method implemented cannot be QPY serialized.
        The subclass must override :meth:`get_waveform` method to return waveform.
        Note that the expression should contain symbol ``t`` that represents a
        sampling time, along with parameters defined in :meth:`.parameters`.
        """
        raise NotImplementedError

    @classmethod
    def _define_constraints(cls) -> List[Union["Expr", List]]:
        """Return a list of symbolic expression for parameter validation.

        A custom pulse without having this method implemented cannot be QPY serialized.
        The subclass must override :meth:`validate_parameters` method to evaluate parameters instead.
        The expressions may contain parameters defined in :attr:`.PARAM_DEF` along with
        ``limit`` that is a class attribute to specify the policy for accepting
        the waveform with the max amplitude exceeding 1.0.

        IF clause can be defined as a list of three expressions representing
        "if", "else", and "then", respectively. This list can be nested.
        """
        raise NotImplementedError

    def get_waveform(self) -> Waveform:
        r"""Return a Waveform with samples filled according to the formula that the pulse
        represents and the parameter values it contains.

        Since the returned array is discretized time series of the continuous function,
        this method uses midpoint sampler. For ``duration``, return:

        .. math::

            \{f(t+0.5) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}

        Returns:
            A waveform representation of this pulse.

        Raises:
            PulseError: When parameters are not bound.
        """
        if self.is_parameterized():
            raise PulseError(
                f"Unassigned parameter exists: {self.parameters}. All parameters should be assigned."
            )

        times = np.arange(0, self.duration) + 1 / 2
        args = (times, *self.param_values)
        waveform = self.envelope(*args)

        return Waveform(samples=waveform, name=self.name)

    def validate_parameters(self) -> None:
        """
        Validate parameters.

        Raises:
            PulseError: If the parameters passed are not valid.
        """
        args = (self.limit_amplitude, *self.param_values)

        for i, constraint in enumerate(self.constraints):
            if isinstance(constraint, list):
                # Cannot use sympy.ITE because it doesn't support lazy evaluation.
                # See https://github.com/sympy/sympy/issues/23295
                eval_res = _evaluate_ite(constraint, *args)
            else:
                eval_res = bool(constraint(*args))
            if not eval_res:
                dict_repr = ", ".join(f"{p} = {v}" for p, v in self.parameters.items())
                const = str(self._define_constraints()[i])
                raise PulseError(
                    f"Assigned parameters {dict_repr} violate following constraint: {const}."
                )

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return any(isinstance(val, ParameterExpression) for val in self.param_values)

    @property
    def parameters(self) -> Dict[str, Any]:
        return dict(zip(self.PARAM_DEF, self.param_values))

    def __eq__(self, other: Pulse) -> bool:
        return super().__eq__(other) and self.parameters == other.parameters

    def __hash__(self) -> int:
        return hash(tuple(self.parameters[k] for k in sorted(self.parameters)))

    def __repr__(self) -> str:
        params_str = ", ".join(f"{p}={v}" for p, v in zip(self.PARAM_DEF, self.param_values))

        return "{}({}{})".format(
            self.__class__.__name__,
            params_str,
            f", name='{self.name}'" if self.name is not None else "",
        )


class Gaussian(SymbolicPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Gaussian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \exp\Bigl( -\frac12 \frac{{(x - \text{duration}/2)}^2}{\text{sigma}^2} \Bigr)\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the gaussian waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedGaussian``, however, for historical
    and practical DSP reasons it has the name ``Gaussian``.
    """
    PARAM_DEF = ["duration", "amp", "sigma"]

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Initialize the gaussian pulse.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the Gaussian envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
        """
        if not isinstance(amp, ParameterExpression):
            amp = complex(amp)

        super().__init__(
            duration=duration,
            parameters=(amp, sigma),
            name=name,
            limit_amplitude=limit_amplitude,
        )

    @classmethod
    def _define_envelope(cls) -> "Expr":
        import sympy as sym

        t, duration, amp, sigma = sym.symbols("t, duration, amp, sigma")
        center = duration / 2
        return amp * _normalized_gaussian(t, center, duration + 2, sigma)

    @classmethod
    def _define_constraints(cls) -> List[Union["Expr", List]]:
        import sympy as sym

        amp, sigma, lim_amp = sym.symbols("amp, sigma, limit")
        return [
            [lim_amp, sym.Abs(amp) <= 1.0, True],
            sigma > 0,
        ]


class GaussianSquare(SymbolicPulse):
    # Not a raw string because we need to be able to split lines.
    """A square pulse with a Gaussian shaped risefall on both sides lifted such that
    its first sample is zero.

    Either the ``risefall_sigma_ratio`` or ``width`` parameter has to be specified.

    If ``risefall_sigma_ratio`` is not None and ``width`` is None:

    .. math::

        \\text{risefall} &= \\text{risefall_sigma_ratio} \\times \\text{sigma}\\\\
        \\text{width} &= \\text{duration} - 2 \\times \\text{risefall}

    If ``width`` is not None and ``risefall_sigma_ratio`` is None:

    .. math:: \\text{risefall} = \\frac{\\text{duration} - \\text{width}}{2}

    In both cases, the lifted gaussian square pulse :math:`f'(x)` is defined as:

    .. math::

        f'(x) &= \\begin{cases}\
            \\exp\\biggl(-\\frac12 \\frac{(x - \\text{risefall})^2}{\\text{sigma}^2}\\biggr)\
                & x < \\text{risefall}\\\\
            1\
                & \\text{risefall} \\le x < \\text{risefall} + \\text{width}\\\\
            \\exp\\biggl(-\\frac12\
                    \\frac{{\\bigl(x - (\\text{risefall} + \\text{width})\\bigr)}^2}\
                          {\\text{sigma}^2}\
                    \\biggr)\
                & \\text{risefall} + \\text{width} \\le x\
        \\end{cases}\\\\
        f(x) &= \\text{amp} \\times \\frac{f'(x) - f'(-1)}{1-f'(-1)},\
            \\quad 0 \\le x < \\text{duration}

    where :math:`f'(x)` is the gaussian square waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedGaussianSquare``, however, for historical
    and practical DSP reasons it has the name ``GaussianSquare``.
    """
    PARAM_DEF = ["duration", "amp", "sigma", "width"]

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        width: Optional[Union[float, ParameterExpression]] = None,
        risefall_sigma_ratio: Optional[Union[float, ParameterExpression]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Initialize the gaussian square pulse.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the Gaussian and of the square pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; see the class
                   docstring for more details.
            width: The duration of the embedded square pulse.
            risefall_sigma_ratio: The ratio of each risefall duration to sigma.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Raises:
            PulseError: When width and risefall_sigma_ratio are both empty.
        """
        if not isinstance(amp, ParameterExpression):
            amp = complex(amp)

        # Convert risefall_sigma_ratio into width which is defined in OpenPulse spec
        if width is None and risefall_sigma_ratio is None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter must be specified."
            )
        if width is not None and risefall_sigma_ratio is not None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter can be specified"
                " but not both."
            )
        if width is None and risefall_sigma_ratio is not None:
            width = duration - 2.0 * risefall_sigma_ratio * sigma

        super().__init__(
            duration=duration,
            parameters=(amp, sigma, width),
            name=name,
            limit_amplitude=limit_amplitude,
        )

    @property
    def risefall_sigma_ratio(self) -> Union[float, ParameterExpression]:
        """Return risefall sigma ratio of the Gaussian rising and falling edges."""
        # (duration - width) / (2.0 sigma)
        return (self.param_values[0] - self.param_values[3]) / (2.0 * self.param_values[2])

    @classmethod
    def _define_envelope(cls) -> "Expr":
        import sympy as sym

        t, duration, amp, sigma, width = sym.symbols("t, duration, amp, sigma, width")
        center = duration / 2

        sq_t0 = center - width / 2
        sq_t1 = center + width / 2
        gaussian_zeroed_width = duration + 2 - width

        gaussian_ledge = _normalized_gaussian(t, sq_t0, gaussian_zeroed_width, sigma)
        gaussian_redge = _normalized_gaussian(t, sq_t1, gaussian_zeroed_width, sigma)

        return amp * sym.Piecewise(
            (gaussian_ledge, t <= sq_t0), (gaussian_redge, t >= sq_t1), (1, True)
        )

    @classmethod
    def _define_constraints(cls) -> List[Union["Expr", List]]:
        import sympy as sym

        duration, amp, sigma, width, lim_amp = sym.symbols("duration, amp, sigma, width, limit")
        return [
            [lim_amp, sym.Abs(amp) <= 1.0, True],
            sigma > 0,
            width >= 0,
            duration >= width,
        ]


class Drag(SymbolicPulse):
    """The Derivative Removal by Adiabatic Gate (DRAG) pulse is a standard Gaussian pulse
    with an additional Gaussian derivative component and lifting applied.

    It is designed to reduce the frequency spectrum of a standard Gaussian pulse near
    the :math:`|1\\rangle\\leftrightarrow|2\\rangle` transition,
    reducing the chance of leakage to the :math:`|2\\rangle` state.

    .. math::

        g(x) &= \\exp\\Bigl(-\\frac12 \\frac{(x - \\text{duration}/2)^2}{\\text{sigma}^2}\\Bigr)\\\\
        g'(x) &= \\text{amp}\\times\\frac{g(x)-g(-1)}{1-g(-1)}\\\\
        f(x) &=  g'(x) \\times \\Bigl(1 + 1j \\times \\text{beta} \\times\
            \\Bigl(-\\frac{x - \\text{duration}/2}{\\text{sigma}^2}\\Bigr)  \\Bigr),
            \\quad 0 \\le x < \\text{duration}

    where :math:`g(x)` is a standard unlifted Gaussian waveform and
    :math:`g'(x)` is the lifted :class:`~qiskit.pulse.library.Gaussian` waveform.

    This pulse, defined by :math:`f(x)`, would be more accurately named as ``LiftedDrag``, however,
    for historical and practical DSP reasons it has the name ``Drag``.

    References:
        1. |citation1|_

        .. _citation1: https://link.aps.org/doi/10.1103/PhysRevA.83.012308

        .. |citation1| replace:: *Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
           Analytic control methods for high-fidelity unitary operations
           in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).*

        2. |citation2|_

        .. _citation2: https://link.aps.org/doi/10.1103/PhysRevLett.103.110501

        .. |citation2| replace:: *F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm
           Phys. Rev. Lett. 103, 110501 – Published 8 September 2009.*
    """

    PARAM_DEF = ["duration", "amp", "sigma", "beta"]

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        beta: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Initialize the drag pulse.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the Drag envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            beta: The correction amplitude.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
        """
        if not isinstance(amp, ParameterExpression):
            amp = complex(amp)

        super().__init__(
            duration=duration,
            parameters=(amp, sigma, beta),
            name=name,
            limit_amplitude=limit_amplitude,
        )

    @classmethod
    def _define_envelope(cls) -> "Expr":
        import sympy as sym

        t, duration, amp, sigma, beta = sym.symbols("t, duration, amp, sigma, beta")
        center = duration / 2

        gauss = amp * _normalized_gaussian(t, center, duration + 2, sigma)
        deriv = -(t - center) / (sigma**2) * gauss

        return gauss + 1j * beta * deriv

    @classmethod
    def _define_constraints(cls) -> List[Union["Expr", List]]:
        import sympy as sym

        t, duration, amp, sigma, beta, lim_amp = sym.symbols("t, duration, amp, sigma, beta, limit")
        drag_eq = cls._define_envelope()
        return [
            [lim_amp, sym.Abs(amp) <= 1.0, True],
            # When large beta
            [
                lim_amp & (sym.Abs(beta) > sigma),
                [
                    # Find the first maxima associated with the beta * d/dx gaussian term
                    # This eq is derived from solving for the roots of the norm of the drag function.
                    # There is a second maxima mirrored around the center of the pulse with the same
                    # norm as the first, so checking the value at the first x maxima is sufficient.
                    duration / 2 - (sigma / beta) * sym.sqrt(beta**2 - sigma**2) >= 0,
                    sym.Abs(
                        drag_eq.subs(
                            [(t, duration / 2 - (sigma / beta) * sym.sqrt(beta**2 - sigma**2))]
                        )
                    )
                    <= 1,
                    sym.Abs(drag_eq.subs([(t, 0)])) <= 1,
                ],
                # If beta <= sigma, then the maximum amplitude is at duration / 2, which is
                # already constrained by amp <= 1
                True,
            ],
            sigma > 0,
            sym.Eq(sym.im(beta), 0),
        ]


class Constant(SymbolicPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    PARAM_DEF = ["duration", "amp"]

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """
        Initialize the constant-valued pulse.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
        """
        if not isinstance(amp, ParameterExpression):
            amp = complex(amp)

        super().__init__(
            duration=duration,
            parameters=(amp,),
            name=name,
            limit_amplitude=limit_amplitude,
        )

    @classmethod
    def _define_envelope(cls) -> "Expr":
        import sympy as sym

        t, duration, amp = sym.symbols("t, duration, amp")

        # Note this is implemented using Piecewise instead of just returning amp
        # directly because otherwise the expression has no t dependence and sympy's
        # lambdify will produce a function f that for an array t returns amp
        # instead of amp * np.ones(t.shape). This does not work well with
        # ParametricPulse.get_waveform().
        #
        # See: https://github.com/sympy/sympy/issues/5642
        return amp * sym.Piecewise((1, (t >= 0) & (t <= duration)), (0, True))

    @classmethod
    def _define_constraints(cls) -> List[Union["Expr", List]]:
        import sympy as sym

        amp, lim_amp = sym.symbols("amp, limit")
        return [
            [lim_amp, sym.Abs(amp) <= 1.0, True],
        ]
