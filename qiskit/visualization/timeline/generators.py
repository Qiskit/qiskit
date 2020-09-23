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

from typing import List, Union, Dict, Any

from qiskit.circuit.exceptions import CircuitError
from qiskit.visualization.timeline import types, drawing_objects


def gen_sched_gate(bit: types.Bits,
                   gate: types.ScheduledGate,
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
        bit: Bit object associated to this drawing.
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
                                          bit=bit,
                                          meta=meta,
                                          styles=styles)
    else:
        # frame change
        styles = {
            'zorder': formatter['layer.frame_change'],
            'color': color,
            'size': formatter['font_size.frame_change'],
            'va': 'center',
            'ha': 'center'
        }
        unicode_symbol = formatter['unicode_symbol.frame_change']
        latex_symbol = formatter['latex_symbol.frame_change']

        drawing = drawing_objects.TextData(data_type=types.DrawingSymbol.FRAME,
                                           bit=bit,
                                           xval=gate.t0,
                                           yval=0,
                                           text=unicode_symbol,
                                           latex=latex_symbol,
                                           styles=styles)

    return [drawing]


def gen_full_gate_name(bit: types.Bits,
                       gate: types.ScheduledGate,
                       formatter: Dict[str, Any]
                       ) -> List[drawing_objects.TextData]:
    """Generate gate name.

    Parameters and associated bits are also shown.

    Stylesheet:
        - `gate_name` style is applied.
        - `gate_latex_repr` key is used to find the latex representation of the gate name.

    Args:
        bit: Bit object associated to this drawing.
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
        'size': formatter['font_size.gate_name'],
        'va': v_align,
        'ha': 'center'
    }
    # find latex representation
    latex_name = formatter.get('gate_latex_repr.{name}'.format(name=gate.operand.name),
                               r'{{\rm {name}}}'.format(name=gate.operand.name))
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

    # assign special name to delay for filtering
    if gate.operand.name == 'delay':
        data_type = types.DrawingLabel.DELAY
    else:
        data_type = types.DrawingLabel.GATE_NAME

    drawing = drawing_objects.TextData(data_type=data_type,
                                       xval=gate.t0 + 0.5 * gate.duration,
                                       yval=v_pos,
                                       bit=bit,
                                       text=label_plain,
                                       latex=label_latex,
                                       styles=styles)

    return [drawing]


def gen_short_gate_name(bit: types.Bits,
                        gate: types.ScheduledGate,
                        formatter: Dict[str, Any]
                        ) -> List[drawing_objects.TextData]:
    """Generate gate name.

    Only operand name is shown.

    Stylesheet:
        - `gate_name` style is applied.
        - `gate_latex_repr` key is used to find the latex representation of the gate name.

    Args:
        bit: Bit object associated to this drawing.
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
        'size': formatter['font_size.gate_name'],
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
                                       bit=bit,
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
        'size': formatter['font_size.bit_name'],
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


def gen_barrier(bit: types.Bits,
                barrier: types.Barrier,
                formatter: Dict[str, Any]
                ) -> List[drawing_objects.LineData]:
    """Generate barrier line.

    Stylesheet:
        - `barrier` style is applied.

    Args:
        bit: Bit object associated to this drawing.
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
                                       bit=bit,
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
