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

"""Symbolic waveform module.

These are pulses which are described by symbolic equations for envelope and parameter constraints.
"""

from typing import Any, Dict, List, Tuple, Optional, Union, Callable, TYPE_CHECKING

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform

if TYPE_CHECKING:
    # Note that the symbolic pulse module doesn't employ symengine due to some missing features.
    # In addition, there are several syntactic difference in boolean expression.
    # Thanks to Lambdify at subclass instantiation, the performance regression is not significant.
    # However, this will still create significant latency for the first sympy import.
    from sympy import Expr, Symbol


def _lifted_gaussian(
    t: "Symbol",
    center: Union["Symbol", "Expr", complex],
    t_zero: Union["Symbol", "Expr", complex],
    sigma: Union["Symbol", "Expr", complex],
) -> "Expr":
    r"""Helper function that returns a lifted Gaussian symbolic equation.

    For :math:`\sigma=` ``sigma`` the symbolic equation will be

    .. math::

        f(x) = \exp\left(-\frac12 \left(\frac{x - \mu}{\sigma}\right)^2 \right),

    with the center :math:`\mu=` ``duration/2``.
    Then, each output sample :math:`y` is modified according to:

    .. math::

        y \mapsto \frac{y-y^*}{1.0-y^*},

    where :math:`y^*` is the value of the un-normalized Gaussian at the endpoints of the pulse.
    This sets the endpoints to :math:`0` while preserving the amplitude at the center,
    i.e. :math:`y` is set to :math:`1.0`.

    Args:
        t: Symbol object representing time.
        center: Symbol or expression representing the middle point of the samples.
        t_zero: The value of t at which the pulse is lowered to 0.
        sigma: Symbol or expression representing Gaussian sigma.

    Returns:
        Symbolic equation.
    """
    import sympy as sym

    gauss = sym.exp(-(((t - center) / sigma) ** 2) / 2)
    offset = sym.exp(-(((t_zero - center) / sigma) ** 2) / 2)

    return (gauss - offset) / (1 - offset)


def _evaluate_ite(ite_lambda_func: Tuple[Union[Callable, Tuple], ...], *values: complex) -> bool:
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
            "This should be a tuple of callables for evaluating if, then, else."
        ) from ex

    if bool(if_func(*values)):
        if isinstance(then_func, tuple):
            # Nested if clause
            return _evaluate_ite(then_func, *values)
        return bool(then_func(*values))
    if isinstance(else_func, tuple):
        # Nested if clause
        return _evaluate_ite(else_func, *values)
    return bool(else_func(*values))


def _lambdify_ite(
    params: List["Symbol"], ite_expr: Tuple["Expr", ...]
) -> Tuple[Union[Callable, List], ...]:
    """Helper function to lambdify ITE-like symbolic expression.

    "ITE" is short for "if, then, else." which is convention of SymPy.

    Args:
        params: Symbols to be used in the expression.
        ite_expr: Symbolic expression which is a tuple of "IF", "THEN", "ELSE".

    Returns:
        Lambdified ITE-like expression.
    """
    import sympy as sym

    lambda_lists = []
    for expr in ite_expr:
        if isinstance(expr, tuple):
            lambda_lists.append(_lambdify_ite(params, expr))
        else:
            lambda_lists.append(sym.lambdify(params, expr))
    return tuple(lambda_lists)


class EnvelopeDescriptor:
    """Descriptor of pulse envelope.

    When new symbolic expression is set for the first time,
    this will internally lambdify the expression and store the callback in the instance cache.
    For the next time it will just return the cached callback for speed up.

    Envelope callback is used to generate waveform samples from the symbolic pulse instance.
    """

    def __init__(self):
        # : Dict[Expr, Callable]
        self.lambda_funcs = dict()

    def __get__(self, instance, owner) -> Callable:
        key = hash(instance.envelope)
        if key not in self.lambda_funcs:
            if instance.envelope is None:
                raise PulseError(f"Envelope of '{instance.pulse_type}' is not assigned.")
            self.__set__(instance, instance.envelope)

        return self.lambda_funcs[key]

    def __set__(self, instance, value):
        key = hash(value)
        if key not in self.lambda_funcs:
            import sympy as sym

            if not isinstance(value, sym.Expr):
                raise PulseError(f"'{repr(value)}' is not a valid symbolic expression.")
            params = [sym.Symbol(p) for p in ["t", "duration", "amp"] + instance._param_names]
            self.lambda_funcs[key] = sym.lambdify(params, value)


class ConstraintsDescriptor:
    """Descriptor of pulse parameter constraints.

    When new symbolic expressions are set for the first time,
    this will internally lambdify the expressions and store the callbacks in the instance cache.
    For the next time it will just return the cached callbacks for speed up.

    Constraints callbacks are used to validate assigned pulse parameters.
    """

    def __init__(self):
        # : Dict[Expr, Callable]
        self.lambda_funcs = dict()

    def __get__(self, instance, owner) -> List[Callable]:
        key = hash(tuple(instance.constraints))
        if key not in self.lambda_funcs:
            self.__set__(instance, instance.constraints)

        return self.lambda_funcs[key]

    def __set__(self, instance, value):
        key = hash(tuple(value))
        if key not in self.lambda_funcs:
            import sympy as sym

            params = [sym.Symbol(p) for p in ["limit", "duration", "amp"] + instance._param_names]
            constraints = []
            for expr in value:
                if isinstance(expr, tuple):
                    # If clause
                    constraints.append(_lambdify_ite(params, expr))
                else:
                    constraints.append(sym.lambdify(params, expr))
            self.lambda_funcs[key] = constraints


class SymbolicPulse(Pulse):
    """The pulse representation model with parameters and symbolic expressions.

    A symbolic pulse instance can be defined with an envelope and parameter constraints.
    Envelope and parameter constraints should be provided with symbolic expressions.
    Rather than creating a subclass, different pulse shapes can be distinguished by
    the instance attributes :attr:`SymbolicPulse.envelope` and :attr:`SymbolicPulse.constraints`,
    together with the ``pulse_type`` argument of the :class:`SymbolicPulse` constructor.


    .. _symbolic_pulse_envelope:

    .. rubric:: Envelope function

    This is defined with an instance attribute :attr:`SymbolicPulse.envelope`
    which can be provided through its setter method.
    The expression must be a function of ``t``, ``duration``, ``amp`` and
    any additional parameters specified for the pulse shape to implement.
    The time ``t`` and ``duration`` are in units of dt, i.e. sample time resolution,
    and this function is sampled with a discrete time vector in [0, ``duration``]
    sampling the pulse envelope at every 0.5 dt (middle sampling strategy) when
    :meth:`SymbolicPulse.get_waveform` method is called.
    The ``amp`` is a complex-valued coefficient that scales the symbolic pulse envelope.
    This indicates conventionally the Qiskit Pulse conforms to the IQ format rather
    than the phasor representation. When a real value is assigned to the ``amp``,
    it is internally typecasted to the complex. The real and imaginary part may be
    directly supplied to two quadratures of the IQ mixer in the control electronics.
    The sample data is not generated until the :meth:`SymbolicPulse.get_waveform` method is called
    thus a symbolic pulse instance only stores parameter values and waveform shape,
    which greatly reduces memory footprint during the program generation.


    .. _symbolic_pulse_constraints:

    .. rubric:: Constraint functions

    Constraints on the parameters are defined with an instance attribute
    :attr:`SymbolicPulse.constraints` which can be provided through its setter method.
    The constraints value must be a list of symbolic expressions, which is a
    function of ``limit`` variable and any additional parameters to be validated
    and must return a boolean value being ``True`` when parameters are valid.
    The symbolic pulse instance can be played only when all constraint functions return ``True``.
    Any number of expressions can be defined. These constraint functions are called when
    a :meth:`SymbolicPulse.validate_parameters` method is called.

    The ``limit`` variable represents the policy of forbidding the maximum amplitude
    from exceeding 1.0 (its default value is ``True``, i.e. it doesn't allow amplitude > 1.0).
    Note that in Qiskit the pulse envelope is represented by complex samples.
    Strictly speaking, the maximum amplitude indicates the maximum norm of the complex values.

    When branching logic (if clause) is necessary, one can use `SymPy ITE`_ class
    or a tuple of expressions consisting of expressions for "IF", "THEN", and "ELSE".
    Typically, the list of constraints includes conditions to ensure the maximum pulse
    amplitude doesn't exceed the signal dynamic range [-1.0, 1.0].
    When validation is not necessary, this can be an empty list.

    .. _SymPy ITE: https://docs.sympy.org/latest/modules/logic.html#sympy.logic.boolalg.ITE


    .. rubric:: Examples

    This is how user can instantiate symbolic pulse instance.
    In this example, we instantiate a custom `Sawtooth` envelope.

    .. jupyter-execute::

        from qiskit.pulse.library import SymbolicPulse

        my_pulse = SymbolicPulse(
            pulse_type="Sawtooth",
            parameters={"duration": 100, "amp": 0.1, "freq": 0.05},
            name="pulse1",
        )

    Note that :class:`SymbolicPulse` can be instantiated without providing
    the envelope and constraints. However, this instance cannot generate waveforms
    without knowing the envelope definition. Now you need to provide the envelope.

    .. jupyter-execute::

        import sympy

        t, amp, freq = sympy.symbols("t, amp, freq")
        envelope = amp * 2 * (freq * t - sympy.floor(1 / 2 + freq * t))
        my_pulse.envelope = envelope

        my_pulse.draw()

    Likewise, you can define :attr:`SymbolicPulse.constraints` for ``my_pulse``.
    After providing the envelope definition, you can generate the waveform data.
    Note that it would be convenient to define a factory function that automatically
    accomplishes this procedure.

    .. code-block:: python

        def Sawtooth(duration, amp, freq, name):
            instance = SymbolicPulse(
                pulse_type="Sawtooth",
                parameters={"duration": duration, "amp": amp, "freq": freq},
                name=name,
            )

            t, amp, freq = sympy.symbols("t, amp, freq")
            instance.envelope = amp * 2 * (freq * t - sympy.floor(1 / 2 + freq * t))

            return instance

    You can also provide a :class:`Parameter` object in the ``parameters`` dictionary
    when you instantiate the symbolic pulse instance. Waveform cannot be
    generated until you assign all unbounded parameters.
    Note that parameters will be assigned through the schedule playing the pulse.


    .. _symbolic_pulse_serialize:

    .. rubric:: Serialization

    The :class:`~SymbolicPulse` subclass is QPY serialized with symbolic expressions.
    A user can therefore create a custom pulse subclass with a novel envelope and constraints,
    and then one can instantiate the class with certain parameters to run on a backend.
    This pulse instance can be saved in the QPY binary, which can be loaded afterwards
    even within the environment not having original class definition loaded.
    This mechanism allows us to easily share a pulse program including custom pulse instructions
    with collaborators or to directly submit the program to the quantum computer backend
    in the parametric form (i.e. not sample data). This greatly reduces amount of data transferred.
    The waveform sample data to be loaded in the control electronics
    will be reconstructed with parameters and deserialized expression objects by the backend.

    .. note::

        Currently QPY serialization of :class:`SymbolicPulse` is not available.
        This feature will be implemented shortly.
        Note that data transmission in the parametric form requires your quantum computer backend
        to support QPY framework with :class:`SymbolicPulse` available. Otherwise, the pulse data
        might be converted into :class:`Waveform`.
    """

    # Lambdify caches keyed on sympy expressions. Returns the corresponding callable.
    _lambda_envelope_cache = EnvelopeDescriptor()
    _lambda_constraints_cache = ConstraintsDescriptor()

    def __init__(
        self,
        pulse_type: str,
        parameters: Dict[str, Union[ParameterExpression, complex]],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create a parametric pulse and validate the input parameters.

        Args:
            pulse_type: Display name of this pulse shape.
            parameters: Dictionary of pulse parameters. This must include "duration".
            name: Display name for this particular pulse envelope.
            limit_amplitude: If ``True``, then limit the absolute value of the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Raises:
            PulseError: When not all parameters are listed in the attribute :attr:`PARAM_DEF`.
        """
        if "duration" not in parameters:
            raise PulseError("'duration' is not defined for this pulse.")
        if "amp" not in parameters:
            raise PulseError("'amp' is not defined for this pulse.")

        super().__init__(
            duration=parameters.pop("duration"),
            name=name,
            limit_amplitude=limit_amplitude,
        )
        amp = parameters.pop("amp")
        if not isinstance(amp, ParameterExpression):
            amp = complex(amp)
        self.amp = amp

        self._pulse_type = pulse_type
        self._param_names = list(parameters.keys())
        self._param_vals = list(parameters.values())
        self._envelope_expr = None
        self._constraint_exprs = []

    def __getattr__(self, item):
        # For backward compatibility. ParametricPulse implements these property methods.

        # Need to use __dict__ property to get instance values inside the __getattr__.
        # Otherwise, run into the maximum recursion error. This class cannot define __slots__.
        defined_params = self.__dict__.get("_param_names", [])
        if item not in defined_params:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

        return self.__dict__["_param_vals"][defined_params.index(item)]

    @property
    def pulse_type(self) -> str:
        """Return display name of the pulse shape."""
        return self._pulse_type

    @property
    def envelope(self) -> "Expr":
        """Return envelope function of the pulse.

        See :ref:`symbolic_pulse_envelope` for details.

        A pulse instance without this value cannot generate waveform samples.
        Note that the expression should contain symbol ``t`` that represents a
        sampling time, along with parameters defined in :meth:`.parameters`.
        """
        return self._envelope_expr

    @envelope.setter
    def envelope(self, new_expr: "Expr"):
        """Add new envelope function to the pulse."""
        self._envelope_expr = new_expr
        self._lambda_envelope_cache = new_expr

    @property
    def constraints(self) -> List["Expr"]:
        """Returns pulse parameter constrains.

        See :ref:`symbolic_pulse_constraints` for details.

        A pulse instance without this value doesn't validate assigned parameters.
        The expressions may contain parameters defined in :attr:`.parameters` along with
        ``limit`` that is a common parameter to specify the policy of forbidding the
        maximum amplitude of the waveform from exceeding 1.0.
        For example, if your waveform defines the parameter ``amp``, this constraint
        could be written as

        .. code-block:: python

            sympy.ITE(limit, sympy.Abs(amp) <= 1.0, sympy.true)

        this expression returns ``amp <= 1.0`` when ``limit`` is set to ``True``,
        otherwise it always returns ``True``, i.e. ``amp`` always passes the validation.
        IF clauses can be also defined as a tuple of three expressions representing
        "if", "else", and "then", respectively.

        .. code-block:: python

            (limit, sympy.Abs(amp) <= 1.0, sympy.true)

        Likewise, you can write any number of expressions for parameters you have defined.
        """
        return self._constraint_exprs

    @constraints.setter
    def constraints(self, new_constraints: List["Expr"]):
        """Add new parameter constraints to the pulse."""
        self._constraint_exprs = new_constraints
        self._lambda_constraints_cache = new_constraints

    def get_waveform(self) -> Waveform:
        r"""Return a Waveform with samples filled according to the formula that the pulse
        represents and the parameter values it contains.

        Since the returned array is a discretized time series of the continuous function,
        this method uses a midpoint sampler. For ``duration``, return:

        .. math::

            \{f(t+0.5) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}

        Returns:
            A waveform representation of this pulse.

        Raises:
            PulseError: When parameters are not bound.
        """
        if self.is_parameterized():
            raise PulseError("Unassigned parameter exists. All parameters must be assigned.")

        times = np.arange(0, self.duration) + 1 / 2
        args = (times, self.duration, self.amp, *self._param_vals)
        waveform = self._lambda_envelope_cache(*args)

        return Waveform(samples=waveform, name=self.name)

    def validate_parameters(self) -> None:
        """Validate parameters.

        Raises:
            PulseError: If the parameters passed are not valid.
        """
        if self.is_parameterized():
            return

        args = (self._limit_amplitude, self.duration, self.amp, *self._param_vals)
        for i, constraint in enumerate(self._lambda_constraints_cache):
            if isinstance(constraint, tuple):
                # Cannot use sympy.ITE because it doesn't support lazy evaluation.
                # See https://github.com/sympy/sympy/issues/23295
                eval_res = _evaluate_ite(constraint, *args)
            else:
                eval_res = bool(constraint(*args))
            if not eval_res:
                param_repr = ", ".join(f"{p}={v}" for p, v in self.parameters.items())
                const_repr = str(self._constraint_exprs[i])
                raise PulseError(
                    f"Assigned parameters {param_repr} violate following constraint: {const_repr}."
                )

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        args = (self.duration, self.amp, *self._param_vals)
        return any(isinstance(val, ParameterExpression) for val in args)

    @property
    def parameters(self) -> Dict[str, Any]:
        params = {"duration": self.duration, "amp": self.amp}
        params.update(dict(zip(self._param_names, self._param_vals)))
        return params

    def __eq__(self, other: "SymbolicPulse") -> bool:
        # Not aware of expressions.
        if not isinstance(other, SymbolicPulse):
            return False
        if self._pulse_type != other._pulse_type:
            # Use pulse type equality rather than class.
            return False
        if self.parameters != other.parameters:
            return False
        return True

    def __hash__(self) -> int:
        return hash(
            (self._pulse_type, self.duration, self.amp, *self._param_names, *self._param_vals)
        )

    def __repr__(self) -> str:
        param_repr = ", ".join(f"{p}={v}" for p, v in self.parameters.items())
        return "{}({}{})".format(
            self._pulse_type,
            param_repr,
            f", name='{self.name}'" if self.name is not None else "",
        )


class Gaussian(SymbolicPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Gaussian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \exp\Bigl( -\frac12 \frac{{(x - \text{duration}/2)}^2}{\text{sigma}^2} \Bigr)\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the gaussian waveform without lifting or amplitude scaling.
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the Gaussian envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        """
        import sympy as sym

        parameters = {
            "duration": duration,
            "amp": amp,
            "sigma": sigma,
        }

        super().__init__(
            pulse_type=self.__class__.__name__,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
        )

        # Add definitions
        t, limit, duration, amp, sigma = sym.symbols("t, limit, duration, amp, sigma")
        center = duration / 2

        self.envelope = amp * _lifted_gaussian(t, center, duration + 1, sigma)
        self.constraints = [
            sym.ITE(limit, sym.Abs(amp) <= 1.0, sym.true),
            sigma > 0,
        ]
        self.validate_parameters()


class GaussianSquare(SymbolicPulse):
    """A square pulse with a Gaussian shaped risefall on both sides lifted such that
    its first sample is zero.

    Exactly one of the ``risefall_sigma_ratio`` and ``width`` parameters has to be specified.

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
    """

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
        """Create new pulse instance.

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
            PulseError: When width and risefall_sigma_ratio are both empty or both non-empty.
        """
        import sympy as sym

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
        else:
            risefall_sigma_ratio = (duration - width) / (2.0 * sigma)

        parameters = {
            "duration": duration,
            "amp": amp,
            "sigma": sigma,
            "width": width,
        }

        super().__init__(
            pulse_type=self.__class__.__name__,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
        )

        # Add definitions
        t, limit, duration, amp, sigma, width = sym.symbols("t, limit, duration, amp, sigma, width")
        center = duration / 2

        sq_t0 = center - width / 2
        sq_t1 = center + width / 2

        gaussian_ledge = _lifted_gaussian(t, sq_t0, -1, sigma)
        gaussian_redge = _lifted_gaussian(t, sq_t1, duration + 1, sigma)

        self.envelope = amp * sym.Piecewise(
            (gaussian_ledge, t <= sq_t0), (gaussian_redge, t >= sq_t1), (1, True)
        )
        self.constraints = [
            sym.ITE(limit, sym.Abs(amp) <= 1.0, sym.true),
            sigma > 0,
            width >= 0,
            duration >= width,
        ]
        self.validate_parameters()

    @property
    def risefall_sigma_ratio(self):
        """Return risefall_sigma_ratio. This is auxiliary parameter to define width."""
        return (self.duration - self.width) / (2.0 * self.sigma)


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

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        beta: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create new pulse instance.

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
        import sympy as sym

        parameters = {
            "duration": duration,
            "amp": amp,
            "sigma": sigma,
            "beta": beta,
        }

        super().__init__(
            pulse_type=self.__class__.__name__,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
        )

        # Add definitions
        t, limit, duration, amp, sigma, beta = sym.symbols("t, limit, duration, amp, sigma, beta")
        center = duration / 2

        gauss = amp * _lifted_gaussian(t, center, duration + 1, sigma)
        deriv = -(t - center) / (sigma**2) * gauss

        drag_eq = gauss + 1j * beta * deriv
        t_drag_max = duration / 2 - (sigma / beta) * sym.sqrt(beta**2 - sigma**2)

        self.envelope = drag_eq
        self.constraints = [
            sym.ITE(limit, sym.Abs(amp) <= 1.0, sym.true),
            # When large beta
            (
                # IF:
                # Avoid using sympy ITE because beta could be approximately zero.
                # Then, evaluation of THEN expression causes zero-division error.
                limit & (sym.Abs(beta) > sigma),
                # THEN:
                # Find the first maxima associated with the beta * d/dx gaussian term
                # This eq is derived from solving for the roots of the norm of the drag function.
                # There is a second maxima mirrored around the center of the pulse with the same
                # norm as the first, so checking the value at the first x maxima is sufficient.
                sym.ITE(
                    t_drag_max > 0,
                    sym.Abs(drag_eq.subs([(t, t_drag_max)])) < 1.0,
                    sym.Abs(drag_eq.subs([(t, 0)])) <= 1.0,
                ),
                # ELSE:
                # When beta <= sigma, then the maximum amplitude is at duration / 2, which is
                # already constrained by amp <= 1
                True,
            ),
            sigma > 0,
            sym.Eq(sym.im(beta), 0),
        ]
        self.validate_parameters()


class Constant(SymbolicPulse):
    """A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
        """
        import sympy as sym

        parameters = {
            "duration": duration,
            "amp": amp,
        }

        super().__init__(
            pulse_type=self.__class__.__name__,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
        )

        # Add definitions
        t, limit, duration, amp = sym.symbols("t, limit, duration, amp")

        # Note this is implemented using Piecewise instead of just returning amp
        # directly because otherwise the expression has no t dependence and sympy's
        # lambdify will produce a function f that for an array t returns amp
        # instead of amp * np.ones(t.shape). This does not work well with
        # ParametricPulse.get_waveform().
        #
        # See: https://github.com/sympy/sympy/issues/5642
        self.envelope = amp * sym.Piecewise((1, (t >= 0) & (t <= duration)), (0, True))
        self.constraints = [sym.ITE(limit, sym.Abs(amp) <= 1.0, sym.true)]
        self.validate_parameters()
