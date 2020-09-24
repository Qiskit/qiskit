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

"""
A collection of functions that generate drawing objects from formatted input data.
See :py:mod:`~qiskit.visualization.timeline.types` for more info on the required data.

An end-user can write arbitrary functions that generate custom drawing objects.
Generators in this module are called with the `formatter` kwarg. This data provides
the stylesheet configuration.


There are 4 types of generators in this module.

1. generator.gates

In this stylesheet entry the input data is `types.ScheduledGate` and generates gate objects
such as time buckets and gate name annotations.

The format of generator is restricted to:

    ```python

    def my_object_generator(
            gate: types.ScheduledGate,
            formatter: Dict[str, Any]) -> List[ElementaryData]:

        # generate gate object.
        pass
    ```

2. generator.bits

In this stylesheet entry the input data is `types.Bits` and generates timeline objects
such as zero line and name of bit associated with the timeline.

The format of generator is restricted to:

    ```python

    def my_object_generator(
            bit: types.Bits,
            formatter: Dict[str, Any]) -> List[ElementaryData]:

        # generate timeline object.
        pass
    ```

3. generator.barriers

In this stylesheet entry the input data is `types.Barrier` and generates barrier objects
such as barrier lines.

The format of generator is restricted to:

    ```python

    def my_object_generator(
            barrier: types.Barrier,
            formatter: Dict[str, Any]) -> List[ElementaryData]:

        # generate barrier object.
        pass
    ```

4. generator.gate_links

In this stylesheet entry the input data is `types.GateLink` and generates barrier objects
such as barrier lines.

The format of generator is restricted to:

    ```python

    def my_object_generator(
            link: types.GateLink,
            formatter: Dict[str, Any]) -> List[ElementaryData]:

        # generate barrier object.
        pass
    ```

Arbitrary generator function satisfying the above format can be accepted.
Returned `ElementaryData` can be arbitrary subclasses that are implemented in
the plotter API.
"""

from typing import List, Union, Dict, Any

from qiskit.circuit.exceptions import CircuitError
from qiskit.visualization.timeline import types, drawing_objects


def gen_sched_gate(gate: types.ScheduledGate,
                   formatter: Dict[str, Any],
                   ) -> List[Union[drawing_objects.TextData, drawing_objects.BoxData]]:
    """Generate time bucket or symbol of scheduled gate.

    If gate duration is zero or frame change a symbol is generated instead of time box.
    The face color of gates depends on the operand type.

    Stylesheet:
        - The `gate` style is applied for finite duration gate.
        - The `frame_change` style is applied for zero duration gate.
        - The `gate_face_color` style is applied for face color.

    Args:
        gate: Gate information source.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `TextData` or `BoxData` drawing objects.
    """
    try:
        unitary = str(gate.operand.to_matrix())
    except (AttributeError, CircuitError):
        unitary = 'n/a'

    try:
        label = gate.operand.label or 'n/a'
    except AttributeError:
        label = 'n/a'

    meta = {
        'name': gate.operand.name,
        'label': label,
        'bits': ', '.join([bit.register.name for bit in gate.bits]),
        't0': gate.t0,
        'duration': gate.duration,
        'unitary': unitary,
        'parameters': ', '.join(map(str, gate.operand.params))
    }

    # find color
    color = formatter.get('gate_face_color.{name}'.format(name=gate.operand.name),
                          formatter['gate_face_color.default'])

    if gate.duration > 0:
        # gate with finite duration pulse
        styles = {
            'zorder': formatter['layer.gate'],
            'facecolor': color,
            'alpha': formatter['alpha.gate'],
            'linewidth': formatter['line_width.gate']
        }

        # assign special name to delay for filtering
        if gate.operand.name == 'delay':
            data_type = types.DrawingBox.DELAY
        else:
            data_type = types.DrawingBox.SCHED_GATE

        drawing = drawing_objects.BoxData(data_type=data_type,
                                          xvals=[gate.t0, gate.t0 + gate.duration],
                                          yvals=[-0.5 * formatter['box_height.gate'],
                                                 0.5 * formatter['box_height.gate']],
                                          bit=gate.bits[gate.bit_position],
                                          meta=meta,
                                          styles=styles)
    else:
        # frame change
        styles = {
            'zorder': formatter['layer.frame_change'],
            'color': color,
            'size': formatter['text_size.frame_change'],
            'va': 'center',
            'ha': 'center'
        }
        unicode_symbol = formatter['unicode_symbol.frame_change']
        latex_symbol = formatter['latex_symbol.frame_change']

        drawing = drawing_objects.TextData(data_type=types.DrawingSymbol.FRAME,
                                           bit=gate.bits[gate.bit_position],
                                           xval=gate.t0,
                                           yval=0,
                                           text=unicode_symbol,
                                           latex=latex_symbol,
                                           styles=styles)

    return [drawing]


def gen_full_gate_name(gate: types.ScheduledGate,
                       formatter: Dict[str, Any]
                       ) -> List[drawing_objects.TextData]:
    """Generate gate name.

    Parameters and associated bits are also shown.

    Stylesheet:
        - `gate_name` style is applied.
        - `gate_latex_repr` key is used to find the latex representation of the gate name.

    Args:
        gate: Gate information source.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `TextData` drawing objects.
    """
    if gate.duration > 0:
        # gate with finite duration pulse
        v_align = 'center'
        v_pos = 0
    else:
        # frame change
        v_align = 'bottom'
        v_pos = formatter['label_offset.frame_change']

    styles = {
        'zorder': formatter['layer.gate_name'],
        'color': formatter['color.gate_name'],
        'size': formatter['text_size.gate_name'],
        'va': v_align,
        'ha': 'center'
    }
    # find latex representation
    latex_name = formatter.get('gate_latex_repr.{name}'.format(name=gate.operand.name),
                               r'{{\rm {name}}}'.format(name=gate.operand.name))

    label_plain = '{name}'.format(name=gate.operand.name)
    label_latex = r'{name}'.format(name=latex_name)

    # bit index
    if len(gate.bits) > 1:
        bits_str = ', '.join(map(str, [bit.index for bit in gate.bits]))
        label_plain += '[{bits}]'.format(bits=bits_str)
        label_latex += '[{bits}]'.format(bits=bits_str)

    # parameter list
    params = []
    for val in gate.operand.params:
        try:
            params.append('{val:.2f}'.format(val=float(val)))
        except ValueError:
            params.append('{val}'.format(val=val))
    params_str = ', '.join(params)

    if params_str and gate.operand.name != 'delay':
        label_plain += '({params})'.format(params=params_str)
        label_latex += '({params})'.format(params=params_str)

    # duration
    if gate.duration > 0:
        label_plain += '[{dur}]'.format(dur=gate.duration)
        label_latex += '[{dur}]'.format(dur=gate.duration)

    # assign special name to delay for filtering
    if gate.operand.name == 'delay':
        data_type = types.DrawingLabel.DELAY
    else:
        data_type = types.DrawingLabel.GATE_NAME

    drawing = drawing_objects.TextData(data_type=data_type,
                                       xval=gate.t0 + 0.5 * gate.duration,
                                       yval=v_pos,
                                       bit=gate.bits[gate.bit_position],
                                       text=label_plain,
                                       latex=label_latex,
                                       styles=styles)

    return [drawing]


def gen_short_gate_name(gate: types.ScheduledGate,
                        formatter: Dict[str, Any]
                        ) -> List[drawing_objects.TextData]:
    """Generate gate name.

    Only operand name is shown.

    Stylesheet:
        - `gate_name` style is applied.
        - `gate_latex_repr` key is used to find the latex representation of the gate name.

    Args:
        gate: Gate information source.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `TextData` drawing objects.
    """
    if gate.duration > 0:
        # gate with finite duration pulse
        v_align = 'center'
        v_pos = 0
    else:
        # frame change
        v_align = 'bottom'
        v_pos = formatter['label_offset.frame_change']

    styles = {
        'zorder': formatter['layer.gate_name'],
        'color': formatter['color.gate_name'],
        'size': formatter['text_size.gate_name'],
        'va': v_align,
        'ha': 'center'
    }
    # find latex representation
    latex_name = formatter.get('gate_latex_repr.{name}'.format(name=gate.operand.name),
                               r'{{\rm {name}}}'.format(name=gate.operand.name))

    label_plain = '{name}'.format(name=gate.operand.name)
    label_latex = '{name}'.format(name=latex_name)

    # assign special name for delay to filtering
    if gate.operand.name == 'delay':
        data_type = types.DrawingLabel.DELAY
    else:
        data_type = types.DrawingLabel.GATE_NAME

    drawing = drawing_objects.TextData(data_type=data_type,
                                       xval=gate.t0 + 0.5 * gate.duration,
                                       yval=v_pos,
                                       bit=gate.bits[gate.bit_position],
                                       text=label_plain,
                                       latex=label_latex,
                                       styles=styles)

    return [drawing]


def gen_timeslot(bit: types.Bits,
                 formatter: Dict[str, Any]
                 ) -> List[drawing_objects.BoxData]:
    """Generate time slot of associated bit.

    Stylesheet:
        - `timeslot` style is applied.

    Args:
        bit: Bit object associated to this drawing.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `TextData` drawing objects.
    """
    styles = {
        'zorder': formatter['layer.timeslot'],
        'alpha': formatter['alpha.timeslot'],
        'linewidth': formatter['line_width.timeslot'],
        'facecolor': formatter['color.timeslot']
    }

    drawing = drawing_objects.BoxData(data_type=types.DrawingBox.TIMELINE,
                                      xvals=[types.AbstractCoordinate.LEFT,
                                             types.AbstractCoordinate.RIGHT],
                                      yvals=[-0.5 * formatter['box_height.timeslot'],
                                             0.5 * formatter['box_height.timeslot']],
                                      bit=bit,
                                      styles=styles)

    return [drawing]


def gen_bit_name(bit: types.Bits,
                 formatter: Dict[str, Any]
                 ) -> List[drawing_objects.TextData]:
    """Generate bit label.

    Stylesheet:
        - `bit_name` style is applied.

    Args:
        bit: Bit object associated to this drawing.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `TextData` drawing objects.
    """
    styles = {
        'zorder': formatter['layer.bit_name'],
        'color': formatter['color.bit_name'],
        'size': formatter['text_size.bit_name'],
        'va': 'center',
        'ha': 'right'
    }

    label_plain = '{name}'.format(name=bit.register.name)
    label_latex = r'{{\rm {register}}}_{{{index}}}'.format(register=bit.register.prefix,
                                                           index=bit.index)

    drawing = drawing_objects.TextData(data_type=types.DrawingLabel.BIT_NAME,
                                       xval=types.AbstractCoordinate.LEFT,
                                       yval=0,
                                       bit=bit,
                                       text=label_plain,
                                       latex=label_latex,
                                       styles=styles)

    return [drawing]


def gen_barrier(barrier: types.Barrier,
                formatter: Dict[str, Any]
                ) -> List[drawing_objects.LineData]:
    """Generate barrier line.

    Stylesheet:
        - `barrier` style is applied.

    Args:
        barrier: Barrier instruction.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `LineData` drawing objects.
    """
    styles = {
        'alpha': formatter['alpha.barrier'],
        'zorder': formatter['layer.barrier'],
        'linewidth': formatter['line_width.barrier'],
        'linestyle': formatter['line_style.barrier'],
        'color': formatter['color.barrier']
    }

    drawing = drawing_objects.LineData(data_type=types.DrawingLine.BARRIER,
                                       xvals=[barrier.t0, barrier.t0],
                                       yvals=[-0.5, 0.5],
                                       bit=barrier.bits[barrier.bit_position],
                                       styles=styles)

    return [drawing]


def gen_bit_link(link: types.GateLink,
                 formatter: Dict[str, Any]
                 ) -> List[drawing_objects.GateLinkData]:
    """Generate bit link line.

    Line color depends on the operand type.

    Stylesheet:
        - `bit_link` style is applied.
        - The `gate_face_color` style is applied for line color.

    Args:
        link: Bit link object.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `LineData` drawing objects.
    """

    # find line color
    color = formatter.get('gate_face_color.{name}'.format(name=link.opname),
                          formatter['gate_face_color.default'])

    styles = {
        'alpha': formatter['alpha.bit_link'],
        'zorder': formatter['layer.bit_link'],
        'linewidth': formatter['line_width.bit_link'],
        'linestyle': formatter['line_style.bit_link'],
        'color': color
    }

    drawing = drawing_objects.GateLinkData(bits=link.bits,
                                           xval=link.t0,
                                           styles=styles)

    return [drawing]
