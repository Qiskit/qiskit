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

# pylint: disable=missing-return-doc

"""Sampler strategy module for sampler functions.

Sampler functions have signature.
    ```python
    def sampler_function(continuous_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
        ...
    ```
where the supplied `continuous_pulse` is a function with signature:
    ```python
    def f(times: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...
    ```
The sampler will call the `continuous_pulse` function with a set of times it will decide
according to the sampling strategy it implements along with the passed `args` and `kwargs`.
"""

from typing import Callable

import numpy as np


def left_sample(continuous_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
    """Left sample a continuous function.

    Args:
        continuous_pulse: Continuous pulse function to sample.
        duration: Duration to sample for.
        *args: Continuous pulse function args.
        **kwargs: Continuous pulse function kwargs.
    """
    times = np.arange(duration)
    return continuous_pulse(times, *args, **kwargs)


def right_sample(continuous_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
    """Sampling strategy for decorator.

    Args:
        continuous_pulse: Continuous pulse function to sample.
        duration: Duration to sample for.
        *args: Continuous pulse function args.
        **kwargs: Continuous pulse function kwargs.
    """
    times = np.arange(1, duration + 1)
    return continuous_pulse(times, *args, **kwargs)


def midpoint_sample(continuous_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
    """Sampling strategy for decorator.

    Args:
        continuous_pulse: Continuous pulse function to sample.
        duration: Duration to sample for.
        *args: Continuous pulse function args.
        **kwargs: Continuous pulse function kwargs.
    """
    times = np.arange(1 / 2, duration + 1 / 2)
    return continuous_pulse(times, *args, **kwargs)
