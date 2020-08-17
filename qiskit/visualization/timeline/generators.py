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

r"""
A collection of functions that generate drawing objects from formatted input data.
See :py:mod:`~qiskit.visualization.timeline.types` for the detail of input data.

Adding a custom generator:
    The function in this module are generators for drawing objects.
    All drawing objects are created by these generators.
    A stylesheet provides a list of generators and the core drawing
    function calls all specified generators for each instruction data.

    An end-user can write arbitrary function with predefined function signature:
        ```python
        def my_object_generator(bit: Union[Qubit, Clbit], gate: ScheduledGate) -> List[TextData]:
            return TextData(
                data_type='user_custom',
                bit=bit,
                x=0,
                y=0,
                text='custom_text')
        ```
    In above example user can add a custom text created by `my_object_generator`.
    This custom generator can be added to the list under the `generator.gate` of the stylesheet.
"""

from typing import List, Union

from qiskit.circuit.exceptions import CircuitError
from qiskit.visualization.timeline import drawer_style, types, drawing_objects


def gen_sched_gate(bit: types.Bits,
                   gate: types.ScheduledGate) \
        -> List[Union[drawing_objects.TextData, drawing_objects.BoxData]]:
    r"""Generate time bucket or symbol of scheduled gate.

    If gate duration is zero or frame change a symbol is generated instead of time box.

    Args:
        bit: Bit object associated to this drawing.
        gate: Gate information source.

    Stylesheet:
        `*.gate` style or `*.frame_change` style is applied.
        The face color depends on the operand type.
        Color is decided by a callback function specified by `layout.gate_color`.
    """
    try:
        unitary = str(gate.operand.to_matrix())
    except (AttributeError, CircuitError):
        unitary = 'n/a'

    try:
        label = gate.operand.label
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

    # gate color table specified by stylesheet
    color_finder = drawer_style['layout.gate_color']

    if gate.duration > 0:
        # gate with finite duration pulse
        styles = {
            'zorder': drawer_style['formatter.layer.gate'],
            'alpha': drawer_style['formatter.alpha.gate'],
            'linewidth': drawer_style['formatter.line_width.gate'],
            'facecolor': color_finder(gate.operand.name)
        }

        # assign special name for delay for filtering
        if gate.operand.name == 'delay':
            data_type = types.DrawingBox.DELAY
        else:
            data_type = types.DrawingBox.SCHED_GATE

        drawing = drawing_objects.BoxData(data_type=data_type,
                                          bit=bit,
                                          x0=gate.t0,
                                          y0=-0.5 * drawer_style['formatter.box_height.gate'],
                                          x1=gate.t0 + gate.duration,
                                          y1=0.5 * drawer_style['formatter.box_height.gate'],
                                          meta=meta,
                                          styles=styles)
    else:
        # frame change
        styles = {
            'zorder': drawer_style['formatter.layer.frame_change'],
            'color': color_finder(gate.operand.name),
            'size': drawer_style['formatter.font_size.frame_change'],
            'va': 'center',
            'ha': 'center'
        }
        unicode_symbol = drawer_style['formatter.unicode_symbol.frame_change']
        latex_symbol = drawer_style['formatter.latex_symbol.frame_change']

        drawing = drawing_objects.TextData(data_type=types.DrawingSymbol.FRAME,
                                           bit=bit,
                                           x=gate.t0,
                                           y=0,
                                           text=unicode_symbol,
                                           latex=latex_symbol,
                                           styles=styles)

    return [drawing]


def gen_full_gate_name(bit: types.Bits,
                       gate: types.ScheduledGate) -> List[drawing_objects.TextData]:
    r"""Generate gate name.

    Parameters and associated bits are also shown.

    Args:
        bit: Bit object associated to this drawing.
        gate: Gate information source.

    Stylesheet:
        `*.gate_name` style is applied.
        Latex gate name is generated by a callback function specified by `layout.latex_gate_name`.
    """
    if gate.duration > 0:
        # gate with finite duration pulse
        v_align = 'center'
        v_pos = 0
    else:
        # frame change
        v_align = 'bottom'
        v_pos = drawer_style['formatter.label_offset.frame_change']

    styles = {
        'zorder': drawer_style['formatter.layer.gate_name'],
        'color': drawer_style['formatter.color.gate_name'],
        'size': drawer_style['formatter.font_size.gate_name'],
        'va': v_align,
        'ha': 'center'
    }
    # gate color table specified by stylesheet
    name_converter = drawer_style['layout.latex_gate_name']

    latex_name = name_converter(gate.operand.name)
    qubits_str = ', '.join([bit.register.name for bit in gate.bits])
    params_str = ', '.join(map(str, gate.operand.params))
    if params_str:
        label_plain = '{name}({qubits})|({params})'.format(name=gate.operand.name,
                                                           qubits=qubits_str,
                                                           params=params_str)
        label_latex = r'{name}_{{\rm {qubits}}}({params})'.format(name=latex_name,
                                                                  qubits=qubits_str,
                                                                  params=params_str)
    else:
        label_plain = '{name}({qubits})'.format(name=gate.operand.name,
                                                qubits=qubits_str)
        label_latex = r'{name}_{{\rm {qubits}}}'.format(name=latex_name,
                                                        qubits=qubits_str)

    # assign special name for delay to filtering
    if gate.operand.name == 'delay':
        data_type = types.DrawingLabel.DELAY
    else:
        data_type = types.DrawingLabel.GATE_NAME

    drawing = drawing_objects.TextData(data_type=data_type,
                                       bit=bit,
                                       x=gate.t0 + 0.5 * gate.duration,
                                       y=v_pos,
                                       text=label_plain,
                                       latex=label_latex,
                                       styles=styles)

    return [drawing]


def gen_short_gate_name(bit: types.Bits,
                        gate: types.ScheduledGate) -> List[drawing_objects.TextData]:
    r"""Generate gate name.

    Only operand name is shown.

    Args:
        bit: Bit object associated to this drawing.
        gate: Gate information source.

    Stylesheet:
        `*.gate_name` style is applied.
        Latex gate name is generated by a callback function specified by `layout.latex_gate_name`.
    """
    if gate.duration > 0:
        # gate with finite duration pulse
        v_align = 'center'
        v_pos = 0
    else:
        # frame change
        v_align = 'bottom'
        v_pos = drawer_style['formatter.label_offset.frame_change']

    styles = {
        'zorder': drawer_style['formatter.layer.gate_name'],
        'color': drawer_style['formatter.color.gate_name'],
        'size': drawer_style['formatter.font_size.gate_name'],
        'va': v_align,
        'ha': 'center'
    }
    # gate color table specified by stylesheet
    name_converter = drawer_style['layout.latex_gate_name']

    label_plain = '{name}'.format(name=gate.operand.name)
    label_latex = '{name}'.format(name=name_converter(gate.operand.name))

    # assign special name for delay to filtering
    if gate.operand.name == 'delay':
        data_type = types.DrawingLabel.DELAY
    else:
        data_type = types.DrawingLabel.GATE_NAME

    drawing = drawing_objects.TextData(data_type=data_type,
                                       bit=bit,
                                       x=gate.t0 + 0.5 * gate.duration,
                                       y=v_pos,
                                       text=label_plain,
                                       latex=label_latex,
                                       styles=styles)

    return [drawing]


def gen_timeslot(bit: types.Bits) -> List[drawing_objects.BoxData]:
    r"""Generate time slot of associated bit.

    Args:
        bit: Bit object associated to this drawing.

    Stylesheet:
        `*.timeslot` style is applied.
    """
    styles = {
        'zorder': drawer_style['formatter.layer.timeslot'],
        'alpha': drawer_style['formatter.alpha.timeslot'],
        'linewidth': drawer_style['formatter.line_width.timeslot'],
        'facecolor': drawer_style['formatter.color.timeslot']
    }

    drawing = drawing_objects.BoxData(data_type=types.DrawingBox.TIMELINE,
                                      bit=bit,
                                      x0=types.AbstractCoordinate.LEFT,
                                      y0=-0.5 * drawer_style['formatter.box_height.timeslot'],
                                      x1=types.AbstractCoordinate.RIGHT,
                                      y1=0.5 * drawer_style['formatter.box_height.timeslot'],
                                      styles=styles)

    return [drawing]


def gen_bit_name(bit: types.Bits) -> List[drawing_objects.TextData]:
    r"""Generate bit label.

    Args:
        bit: Bit object associated to this drawing.

    Stylesheet:
        `*.bit_name` style is applied.
    """
    styles = {
        'zorder': drawer_style['formatter.layer.bit_name'],
        'color': drawer_style['formatter.color.bit_name'],
        'size': drawer_style['formatter.font_size.bit_name'],
        'va': 'center',
        'ha': 'right'
    }

    label_plain = '{name}'.format(name=bit.register.name)
    label_latex = r'{{\rm {register}}}_{{{index}}}'.format(register=bit.register.prefix,
                                                           index=bit.index)

    drawing = drawing_objects.TextData(data_type=types.DrawingLabel.BIT_NAME,
                                       bit=bit,
                                       x=types.AbstractCoordinate.LEFT,
                                       y=0,
                                       text=label_plain,
                                       latex=label_latex,
                                       styles=styles)

    return [drawing]


def gen_barrier(bit: types.Bits,
                barrier: types.Barrier) -> List[drawing_objects.LineData]:
    r"""Generate barrier line.

    Args:
        bit: Bit object associated to this drawing.
        barrier: Barrier instruction.

    Stylesheet:
        `*.barrier` style is applied.
    """
    styles = {
        'alpha': drawer_style['formatter.alpha.barrier'],
        'zorder': drawer_style['formatter.layer.barrier'],
        'linewidth': drawer_style['formatter.line_width.barrier'],
        'linestyle': drawer_style['formatter.line_style.barrier'],
        'color': drawer_style['formatter.color.barrier']
    }

    drawing = drawing_objects.LineData(data_type=types.DrawingLine.BARRIER,
                                       bit=bit,
                                       x=[barrier.t0, barrier.t0],
                                       y=[-0.5, 0.5],
                                       styles=styles)

    return [drawing]


def gen_bit_link(link: types.GateLink) -> List[drawing_objects.BitLinkData]:
    r"""Generate bit link line.

    Args:
        link: Bit link object.

    Stylesheet:
        `*.bit_link` style is applied.
        The line color depends on the operand type.
        Color is decided by a callback function specified by `layout.gate_color`.
    """

    # gate color table specified by stylesheet
    color_finder = drawer_style['layout.gate_color']

    styles = {
        'alpha': drawer_style['formatter.alpha.bit_link'],
        'zorder': drawer_style['formatter.layer.bit_link'],
        'linewidth': drawer_style['formatter.line_width.bit_link'],
        'linestyle': drawer_style['formatter.line_style.bit_link'],
        'color': color_finder(link.operand.name)
    }

    drawing = drawing_objects.BitLinkData(bits=link.bits,
                                          x=link.t0,
                                          styles=styles)

    return [drawing]
