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
Stylesheet for pulse drawer.

# TODO update docstring

The stylesheet `QiskitPulseStyle` is initialized with the hard-corded default values in
`default_style`. This instance is generated when the pulse drawer module is loaded so that
every lower modules can access to the information.

The `QiskitPulseStyle` is a wrapper class of python dictionary with the structured keys
such as `formatter.color.fill_waveform_d` to represent a color code of the drive channel.
This key representation and initialization framework are the imitative of
`rcParams` of `matplotlib`. However, the `QiskitPulseStyle` is not compatible with the `rcParams`
because the pulse stylesheet is heavily specialized to the context of the pulse program.

The settings of stylesheet are broadly separated into `formatter` and `generator`.

Formatter
~~~~~~~~~
The `formatter` is a dictionary of drawing parameters to control the appearance of
visualization elements. This data structure is similar to the `rcParams`.
An end-user can add custom keys that are consumed by the user-defined generators.

General setting (any):
- `formatter.general.fig_size`: Tuple represents the figure size.
- `formatter.general.dpi`: An integer represents the image resolution.

Color (str, html color code):
- `formatter.color.fill_waveform_d`: `ComplexColors` instance represents
    the color of waveform in drive channels.
- `formatter.color.fill_waveform_u`: `ComplexColors` instance represents
    the color of waveform in control channels.
- `formatter.color.fill_waveform_m`: `ComplexColors` instance represents
    the color of waveform in measure channels.
- `formatter.color.fill_waveform_a`: `ComplexColors` instance represents
    the color of waveform in acquire channels.
- `formatter.color.baseline`: Color code for baselines.
- `formatter.color.barrier`: Color code for barriers.
- `formatter.color.background`: Color code for the canvas.
- `formatter.color.annotate`: Color code for annotations.
- `formatter.color.frame_change`: Color code for frame change information.
- `formatter.color.snapshot`: Color code for snapshot information.
- `formatter.color.axis_label`: Color code for axis labels.

Transparency (float value from 0 to 1):
- formatter.alpha.fill_waveform`: Transparency of filled waveforms.
- formatter.alpha.baseline`: Transparency of baselines.
- formatter.alpha.barrier`: Transparency of barriers.

Layer (integer, larger values come front):
- formatter.layer.fill_waveform`: Layer position of filled waveforms.
- formatter.layer.baseline`: Layer position of baselines.
- formatter.layer.barrier`: Layer position of barriers.
- formatter.layer.annotate`: Layer position of annotations.
- formatter.layer.axis_label`: Layer position of axis labels.
- formatter.layer.frame_change`: Layer position of frame change information.
- formatter.layer.snapshot`: Layer position of snapshot information.

Margins (float):
- `formatter.margin.top`: Top margin of the canvas in units of pulse height.
- `formatter.margin.bottom`: Bottom margin of the canvas in units of pulse height.
- `formatter.margin.left`: Left margin of the canvas in units of schedule length.
- `formatter.margin.right`: Right margin of the canvas in units of schedule length.
- `formatter.margin.between_channel`: Spacing between channels in units of pulse height.

Label offset (float):
- `formatter.label_offset.pulse_name`: Offset of pulse name labels from the baseline in
    units of pulse height.
- `formatter.label_offset.scale_factor`: Offset of channel's scale factor from the baseline in
    units of pulse height.
- `formatter.label_offset.frame_change`: Offset of frame labels from the baseline in
    units of pulse height.
- `formatter.label_offset.snapshot`: Offset of snapshot labels from the baseline in
    units of pulse height.

Text size (float)
- `formatter.text_size.axis_label`: Text size of axis labels.
- `formatter.text_size.annotate`: Text size of annotations.
- `formatter.text_size.frame_change`: Symbol character size of frame change.
- `formatter.text_size.snapshot`: Symbol character size of snapshot.
- `formatter.text_size.fig_title`: Text size of the figure title.

Line width (float):
- `formatter.line_width.fill_waveform`: Line width of filled waveforms.
- `formatter.line_width.baseline`: Line width of baselines.
- `formatter.line_width.barrier`: Line width of barriers.

Line style (str, the syntax conforms to `matplotlib`)
- `formatter.line_style.fill_waveform`: Line style of filled waveforms.
- `formatter.line_style.baseline`: Line style of baselines.
- `formatter.line_style.barrier`: Line style of barriers.

Control (bool)
- `formatter.control.apply_phase_modulation`: Set `True` to apply a phase factor
    to waveform envelopes.
- `formatter.control.show_snapshot_channel`: Set `True` to show the snapshot channel.
- `formatter.control.show_acquire_channel`: Set `True` to show acquire channels.
- `formatter.control.show_empty_channel`: Set `True` to show no waveform channels.

Unicode symbol (str)
- `formatter.unicode_symbol.frame_change`: Unicode expression of frame change.
- `formatter.unicode_symbol.snapshot`: Unicode expression of snapshot.

Latex symbol (str)
- `formatter.latex_symbol.frame_change`: Latex expression of frame change.
- `formatter.latex_symbol.snapshot`: Latex expression of snapshot.

Generator
~~~~~~~~~
The `generator` is a collection of callback functions to generate drawing objects.
An end-user can add custom functions to draw user-defined drawing objects.
See py:mod:`qiskit.visualization.pulse_v2.generators` for the detail of generators.

- `generator.waveform`: Generate drawing objects from waveform type instructions.
- `generator.frame`: Generate drawing objects from frame type instructions.
- `generator.channel`: Generate drawing objects from channel information.
- `generator.snapshot`: Generate drawing objects from snapshot information.
- `generator.barrier`: Generate drawing objects from barrier information.
"""

import warnings
from typing import Dict, Any, Mapping
from qiskit.visualization.pulse_v2 import generators, layouts


class QiskitPulseStyle(dict):
    """Stylesheet for pulse drawer."""
    _deprecated_keys = {}

    def __init__(self):
        super().__init__()
        # to inform which stylesheet is applied. some plotter may not support specific style.
        self.stylesheet = None
        self.update(default_style())

    def update(self, __m: Mapping[str, Any], **kwargs) -> None:
        super().update(__m, **kwargs)
        for key, value in __m.items():
            if key in self._deprecated_keys:
                warnings.warn('%s is deprecated. Use %s instead.'
                              % (key, self._deprecated_keys[key]),
                              DeprecationWarning)
                self.__setitem__(self._deprecated_keys[key], value)
            else:
                self.__setitem__(key, value)
        self.stylesheet = __m.__class__.__name__

    @property
    def formatter(self):
        """Return formatter field of style dictionary."""
        sub_dict = dict()
        for key, value in self.items():
            sub_keys = key.split('.')
            if sub_keys[0] == 'formatter':
                sub_dict['.'.join(sub_keys[1:])] = value
        return sub_dict

    @property
    def generator(self):
        """Return generator field of style dictionary."""
        sub_dict = dict()
        for key, value in self.items():
            sub_keys = key.split('.')
            if sub_keys[0] == 'generator':
                sub_dict['.'.join(sub_keys[1:])] = value
        return sub_dict

    @property
    def layout(self):
        """Return layout field of style dictionary."""
        sub_dict = dict()
        for key, value in self.items():
            sub_keys = key.split('.')
            if sub_keys[0] == 'layout':
                sub_dict['.'.join(sub_keys[1:])] = value
        return sub_dict


class IqxStandard(dict):
    """Standard pulse stylesheet.

    - Generate stepwise waveform envelope with latex pulse names.
    - Apply phase modulation to waveforms.
    - Plot frame change symbol with formatted operand values.
    - Show chart name with scaling factor.
    - Show snapshot and barrier.
    - Channels are sorted by index and control channels are added to the end.
    """
    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.apply_phase_modulation': True,
                 'formatter.control.show_snapshot_channel': True,
                 'formatter.control.show_empty_channel': False,
                 'formatter.control.auto_chart_scaling': True,
                 'formatter.control.axis_break': True,
                 'generator.waveform': [generators.gen_filled_waveform_stepwise,
                                        generators.gen_ibmq_latex_waveform_name],
                 'generator.frame': [generators.gen_frame_symbol,
                                     generators.gen_formatted_frame_values],
                 'generator.chart': [generators.gen_chart_name,
                                     generators.gen_baseline,
                                     generators.gen_channel_freqs],
                 'generator.snapshot': [generators.gen_snapshot_symbol],
                 'generator.barrier': [generators.gen_barrier],
                 'layout.chart_channel_map': layouts.channel_index_grouped_sort_u,
                 'layout.time_axis_map': layouts.time_map_in_ns}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Standard Pulse style sheet.'


class IqxPublication(dict):
    """Simple pulse stylesheet suited for publication.

    - Generate stepwise waveform envelope with latex pulse names.
    - Apply phase modulation to waveforms.
    - Do not show frame changes.
    - Show chart name.
    - Do not show snapshot and barrier.
    - Channels are sorted by qubit index.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.apply_phase_modulation': True,
                 'formatter.control.show_snapshot_channel': True,
                 'formatter.control.show_empty_channel': False,
                 'formatter.control.auto_chart_scaling': False,
                 'formatter.control.axis_break': True,
                 'formatter.channel_scaling.drive': 5.0,
                 'formatter.channel_scaling.measure': 5.0,
                 'generator.waveform': [generators.gen_filled_waveform_stepwise,
                                        generators.gen_ibmq_latex_waveform_name],
                 'generator.frame': [],
                 'generator.chart': [generators.gen_chart_name,
                                     generators.gen_baseline],
                 'generator.snapshot': [],
                 'generator.barrier': [],
                 'layout.chart_channel_map': layouts.qubit_index_sort,
                 'layout.time_axis_map': layouts.time_map_in_ns}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Simple pulse style sheet for publication.'


class IqxDebugging(dict):
    """Pulse stylesheet for pulse programmers. Show details of instructions.

    # TODO: add more generators

    - Generate stepwise waveform envelope with latex pulse names.
    - Generate annotation for waveform height.
    - Do not apply phase modulation to waveforms.
    - Plot frame change symbol with raw operand values.
    - Show chart name and channel frequency.
    - Show snapshot and barrier.
    - Channels are sorted by index and control channels are added to the end.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.apply_phase_modulation': False,
                 'formatter.control.show_snapshot_channel': True,
                 'formatter.control.show_empty_channel': False,
                 'formatter.control.auto_chart_scaling': False,
                 'formatter.control.axis_break': True,
                 'generator.waveform': [generators.gen_filled_waveform_stepwise,
                                        generators.gen_ibmq_latex_waveform_name,
                                        generators.gen_waveform_max_value],
                 'generator.frame': [generators.gen_frame_symbol,
                                     generators.gen_raw_operand_values_compact],
                 'generator.chart': [generators.gen_chart_name,
                                     generators.gen_baseline,
                                     generators.gen_channel_freqs],
                 'generator.snapshot': [generators.gen_snapshot_symbol,
                                        generators.gen_snapshot_name],
                 'generator.barrier': [generators.gen_barrier],
                 'layout.chart_channel_map': layouts.channel_index_grouped_sort_u,
                 'layout.time_axis_map': layouts.time_map_in_ns}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Pulse style sheet for pulse programmers.'


def default_style() -> Dict[str, Any]:
    """Define default values of the pulse stylesheet."""
    return {
        'formatter.general.fig_width': 13,
        'formatter.general.fig_chart_height': 1.5,
        'formatter.general.dpi': 150,
        'formatter.general.vertical_resolution': 1e-6,
        'formatter.general.max_scale': 100,
        'formatter.color.fill_waveform_w': ['#648fff', '#002999'],
        'formatter.color.fill_waveform_d': ['#648fff', '#002999'],
        'formatter.color.fill_waveform_u': ['#ffb000', '#994A00'],
        'formatter.color.fill_waveform_m': ['#dc267f', '#760019'],
        'formatter.color.fill_waveform_a': ['#dc267f', '#760019'],
        'formatter.color.baseline': '#000000',
        'formatter.color.barrier': '#222222',
        'formatter.color.background': '#f2f3f4',
        'formatter.color.annotate': '#222222',
        'formatter.color.frame_change': '#000000',
        'formatter.color.snapshot': '#000000',
        'formatter.color.axis_label': '#000000',
        'formatter.alpha.fill_waveform': 0.3,
        'formatter.alpha.baseline': 1.0,
        'formatter.alpha.barrier': 0.7,
        'formatter.layer.fill_waveform': 2,
        'formatter.layer.baseline': 1,
        'formatter.layer.barrier': 1,
        'formatter.layer.annotate': 5,
        'formatter.layer.axis_label': 5,
        'formatter.layer.frame_change': 4,
        'formatter.layer.snapshot': 3,
        'formatter.margin.top': 0.5,
        'formatter.margin.bottom': 0.5,
        'formatter.margin.left_percent': 0.05,
        'formatter.margin.right_percent': 0.05,
        'formatter.margin.between_channel': 0.2,
        'formatter.label_offset.pulse_name': -0.1,
        'formatter.label_offset.scale_factor': -0.15,
        'formatter.label_offset.frame_change': 0.1,
        'formatter.label_offset.snapshot': 0.1,
        'formatter.text_size.axis_label': 15,
        'formatter.text_size.annotate': 12,
        'formatter.text_size.frame_change': 20,
        'formatter.text_size.snapshot': 20,
        'formatter.text_size.fig_title': 15,
        'formatter.text_size.axis_break_symbol': 15,
        'formatter.line_width.fill_waveform': 0,
        'formatter.line_width.axis_break': 6,
        'formatter.line_width.baseline': 1,
        'formatter.line_width.barrier': 1,
        'formatter.line_style.fill_waveform': '-',
        'formatter.line_style.baseline': '-',
        'formatter.line_style.barrier': ':',
        'formatter.channel_scaling.drive': 1.0,
        'formatter.channel_scaling.control': 1.0,
        'formatter.channel_scaling.measure': 1.0,
        'formatter.channel_scaling.acquire': 1.0,
        'formatter.channel_scaling.pos_spacing': 0.1,
        'formatter.channel_scaling.neg_spacing': -0.1,
        'formatter.axis_break.length': 3000,
        'formatter.axis_break.max_length': 1000,
        'formatter.control.apply_phase_modulation': True,
        'formatter.control.show_snapshot_channel': True,
        'formatter.control.show_empty_channel': True,
        'formatter.control.auto_chart_scaling': True,
        'formatter.control.axis_break': True,
        'formatter.unicode_symbol.frame_change': u'\u21BA',
        'formatter.unicode_symbol.snapshot': u'\u21AF',
        'formatter.latex_symbol.frame_change': r'\circlearrowleft',
        'formatter.latex_symbol.snapshot': '',
        'generator.waveform': [],
        'generator.frame': [],
        'generator.chart': [],
        'generator.snapshot': [],
        'generator.barrier': [],
        'layout.chart_channel_map': None,
        'layout.time_axis_map': None}
