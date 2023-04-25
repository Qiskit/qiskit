# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sampler decorator module for sampling of continuous pulses to discrete pulses to be
exposed to user.

Some atypical boilerplate has been added to solve the problem of decorators not preserving
their wrapped function signatures. Below we explain the problem that samplers solve and how
we implement this.

A sampler is a function that takes an continuous pulse function with signature:
    ```python
    def f(times: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...
    ```
and returns a new function:
    def f(duration: int, *args, **kwargs) -> Waveform:
        ...

Samplers are used to build up pulse waveforms from continuous pulse functions.

In Python the creation of a dynamic function that wraps another function will cause
the underlying signature and documentation of the underlying function to be overwritten.
In order to circumvent this issue the Python standard library provides the decorator
`functools.wraps` which allows the programmer to expose the names and signature of the
wrapped function as those of the dynamic function.

Samplers are implemented by creating a function with signature
    @sampler
    def left(continuous_pulse: Callable, duration: int, *args, **kwargs)
        ...

This will create a sampler function for `left`. Since it is a dynamic function it would not
have the docstring of `left` available too `help`. This could be fixed by wrapping with
`functools.wraps` in the `sampler`, but this would then cause the signature to be that of the
sampler function which is called on the continuous pulse, below:
    `(continuous_pulse: Callable, duration: int, *args, **kwargs)``
This is not correct for the sampler as the output sampled functions accept only a function.
For the standard sampler we get around this by not using `functools.wraps` and
explicitly defining our samplers such as `left`, `right` and `midpoint` and
calling `sampler` internally on the function that implements the sampling schemes such as
`left_sample`, `right_sample` and `midpoint_sample` respectively. See `left` for an example of this.


In this way our standard samplers will expose the proper help signature, but a user can
still create their own sampler with
    @sampler
    def custom_sampler(time, *args, **kwargs):
        ...
However, in this case it will be missing documentation of the underlying sampling methods.
We believe that the definition of custom samplers will be rather infrequent.

However, users will frequently apply sampler instances too continuous pulses. Therefore, a different
approach was required for sampled continuous functions (the output of an continuous pulse function
decorated by a sampler instance).

A sampler instance is a decorator that may be used to wrap continuous pulse functions such as
linear below:
```python
    @left
    def linear(times: np.ndarray, m: float, b: float) -> np.ndarray:
        ```Linear test function
        Args:
            times: Input times.
            m: Slope.
            b: Intercept
        Returns:
            np.ndarray
        ```
        return m*times+b
```
Which after decoration may be called with a duration rather than an array of times
    ```python
    duration = 10
    pulse_envelope = linear(10, 0.1, 0.1)
    ```
If one calls help on `linear` they will find
    ```
    linear(duration:int, *args, **kwargs) -> numpy.ndarray
    Discretized continuous pulse function: `linear` using
    sampler: `_left`.

     The first argument (time) of the continuous pulse function has been replaced with
     a discretized `duration` of type (int).

     Args:
         duration (int)
         *args: Remaining arguments of continuous pulse function.
                See continuous pulse function documentation below.
         **kwargs: Remaining kwargs of continuous pulse function.
                   See continuous pulse function documentation below.

     Sampled continuous function:

        function linear in module test.python.pulse.test_samplers
        linear(x:numpy.ndarray, m:float, b:float) -> numpy.ndarray
            Linear test function
            Args:
                x: Input times.
                m: Slope.
                b: Intercept
            Returns:
                np.ndarray
    ```
This is partly because `functools.wraps` has been used on the underlying function.
This in itself is not sufficient as the signature of the sampled function has
`duration`, whereas the signature of the continuous function is `time`.

This is achieved by removing `__wrapped__` set by `functools.wraps` in order to preserve
the correct signature and also applying `_update_annotations` and `_update_docstring`
to the generated function which corrects the function annotations and adds an informative
docstring respectively.

The user therefore has access to the correct sampled function docstring in its entirety, while
still seeing the signature for the continuous pulse function and all of its arguments.
"""

import functools
from typing import Callable
import textwrap
import pydoc

import numpy as np

from ...exceptions import PulseError
from ..waveform import Waveform
from . import strategies


def functional_pulse(func: Callable) -> Callable:
    """A decorator for generating Waveform from python callable.

    Args:
        func: A function describing pulse envelope.

    Raises:
        PulseError: when invalid function is specified.
    """

    @functools.wraps(func)
    def to_pulse(duration, *args, name=None, **kwargs):
        """Return Waveform."""
        if isinstance(duration, (int, np.integer)) and duration > 0:
            samples = func(duration, *args, **kwargs)
            samples = np.asarray(samples, dtype=np.complex128)
            return Waveform(samples=samples, name=name)
        raise PulseError("The first argument must be an integer value representing duration.")

    return to_pulse


def _update_annotations(discretized_pulse: Callable) -> Callable:
    """Update annotations of discretized continuous pulse function with duration.

    Args:
        discretized_pulse: Discretized decorated continuous pulse.
    """
    undecorated_annotations = list(discretized_pulse.__annotations__.items())
    decorated_annotations = undecorated_annotations[1:]
    decorated_annotations.insert(0, ("duration", int))
    discretized_pulse.__annotations__ = dict(decorated_annotations)
    return discretized_pulse


def _update_docstring(discretized_pulse: Callable, sampler_inst: Callable) -> Callable:
    """Update annotations of discretized continuous pulse function.

    Args:
        discretized_pulse: Discretized decorated continuous pulse.
        sampler_inst: Applied sampler.
    """
    wrapped_docstring = pydoc.render_doc(discretized_pulse, "%s")
    header, body = wrapped_docstring.split("\n", 1)
    body = textwrap.indent(body, "                    ")
    wrapped_docstring = header + body
    updated_ds = """
                Discretized continuous pulse function: `{continuous_name}` using
                sampler: `{sampler_name}`.

                 The first argument (time) of the continuous pulse function has been replaced with
                 a discretized `duration` of type (int).

                 Args:
                     duration (int)
                     *args: Remaining arguments of continuous pulse function.
                            See continuous pulse function documentation below.
                     **kwargs: Remaining kwargs of continuous pulse function.
                               See continuous pulse function documentation below.

                 Sampled continuous function:

                    {continuous_doc}
                """.format(
        continuous_name=discretized_pulse.__name__,
        sampler_name=sampler_inst.__name__,
        continuous_doc=wrapped_docstring,
    )

    discretized_pulse.__doc__ = updated_ds
    return discretized_pulse


def sampler(sample_function: Callable) -> Callable:
    """Sampler decorator base method.

    Samplers are used for converting an continuous function to a discretized pulse.

    They operate on a function with the signature:
        `def f(times: np.ndarray, *args, **kwargs) -> np.ndarray`
    Where `times` is a numpy array of floats with length n_times and the output array
    is a complex numpy array with length n_times. The output of the decorator is an
    instance of `FunctionalPulse` with signature:
        `def g(duration: int, *args, **kwargs) -> Waveform`

    Note if your continuous pulse function outputs a `complex` scalar rather than a
    `np.ndarray`, you should first vectorize it before applying a sampler.


    This class implements the sampler boilerplate for the sampler.

    Args:
        sample_function: A sampler function to be decorated.
    """

    def generate_sampler(continuous_pulse: Callable) -> Callable:
        """Return a decorated sampler function."""

        @functools.wraps(continuous_pulse)
        def call_sampler(duration: int, *args, **kwargs) -> Waveform:
            """Replace the call to the continuous function with a call to the sampler applied
            to the analytic pulse function."""
            sampled_pulse = sample_function(continuous_pulse, duration, *args, **kwargs)
            return np.asarray(sampled_pulse, dtype=np.complex_)

        # Update type annotations for wrapped continuous function to be discrete
        call_sampler = _update_annotations(call_sampler)
        # Update docstring with that of the sampler and include sampled function documentation.
        call_sampler = _update_docstring(call_sampler, sample_function)
        # Unset wrapped to return base sampler signature
        # but still get rest of benefits of wraps
        # such as __name__, __qualname__
        call_sampler.__dict__.pop("__wrapped__")
        # wrap with functional pulse
        return functional_pulse(call_sampler)

    return generate_sampler


def left(continuous_pulse: Callable) -> Callable:
    r"""Left sampling strategy decorator.

    See `pulse.samplers.sampler` for more information.

    For `duration`, return:
        $$\{f(t) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}$$

    Args:
        continuous_pulse: To sample.
    """

    return sampler(strategies.left_sample)(continuous_pulse)


def right(continuous_pulse: Callable) -> Callable:
    r"""Right sampling strategy decorator.

    See `pulse.samplers.sampler` for more information.

    For `duration`, return:
        $$\{f(t) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<t<=\texttt{duration}\}$$

    Args:
        continuous_pulse: To sample.
    """

    return sampler(strategies.right_sample)(continuous_pulse)


def midpoint(continuous_pulse: Callable) -> Callable:
    r"""Midpoint sampling strategy decorator.

    See `pulse.samplers.sampler` for more information.

    For `duration`, return:
        $$\{f(t+0.5) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}$$

    Args:
        continuous_pulse: To sample.
    """
    return sampler(strategies.midpoint_sample)(continuous_pulse)
