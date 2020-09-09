# -*- coding: utf-8 -*-

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

"""Barrier generators.

A collection of functions that generate drawing object for input input barrier type instructions.
See py:mod:`qiskit.visualization.pulse_v2.types` for the detail of input data.

In this module input data is `BarrierInstruction`.

An end-user can write arbitrary functions that generate custom drawing objects.
Generators in this module are called with `formatter` and `device` kwargs.
These data provides stylesheet configuration and backend system configuration.

The format of generator is restricted to:

    ```python

    def my_object_generator(data: BarrierInstruction,
                            formatter: Dict[str, Any],
                            device: DrawerBackendInfo) -> List[ElementaryData]:
        pass
    ```

Arbitrary generator function satisfying above format can be accepted.
Returned `ElementaryData` can be arbitrary subclass that is implemented in plotter API.

"""
from typing import Dict, Any, List

from qiskit.visualization.pulse_v2 import drawing_objects, types, device_info


def gen_barrier(data: types.BarrierInstruction,
                formatter: Dict[str, Any],
                device: device_info.DrawerBackendInfo) \
        -> List[drawing_objects.LineData]:
    """Generate a barrier from provided relative barrier instruction..

    Stylesheets:
        - The `barrier` style is applied.

    Args:
        data: Barrier instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.
    Returns:
        List of `LineData` drawing objects.
    """
    style = {'alpha': formatter['alpha.barrier'],
             'zorder': formatter['layer.barrier'],
             'linewidth': formatter['line_width.barrier'],
             'linestyle': formatter['line_style.barrier'],
             'color': formatter['color.barrier']}

    line = drawing_objects.LineData(data_type=types.DrawingLine.BARRIER,
                                    channels=data.channels,
                                    xvals=[data.t0, data.t0],
                                    yvals=[-1, 1],
                                    styles=style)

    return [line]
