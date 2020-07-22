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
Stylesheet for pulse drawer.

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


def default_style() -> Dict[str, Any]:
    """Define default values of the pulse stylesheet."""
    return {
        'formatter.general.fig_size': [8, 6],
        'formatter.general.dpi': 150,
        'formatter.color.fill_waveform_d': ['#648fff', '#002999'],
        'formatter.color.fill_waveform_u': ['#ffb000', '#994A00'],
        'formatter.color.fill_waveform_m': ['#dc267f', '#760019'],
        'formatter.color.fill_waveform_a': ['#dc267f', '#760019'],
        'formatter.color.baseline': '#000000',
        'formatter.color.barrier': '#222222',
        'formatter.color.background': 'f2f3f4',
        'formatter.color.annotate': '#222222',
        'formatter.color.frame_change': '#000000',
        'formatter.color.snapshot': '#000000',
        'formatter.color.axis_label': '#000000',
        'formatter.alpha.fill_waveform': 1.0,
        'formatter.alpha.baseline': 1.0,
        'formatter.alpha.barrier': 0.7,
        'formatter.layer.fill_waveform': 2,
        'formatter.layer.baseline': 1,
        'formatter.layer.barrier': 1,
        'formatter.layer.annotate': 4,
        'formatter.layer.axis_label': 4,
        'formatter.layer.frame_change': 3,
        'formatter.layer.snapshot': 3,
        'formatter.margin.top': 0.2,
        'formatter.margin.bottom': 0.2,
        'formatter.margin.left': 0.05,
        'formatter.margin.right': 0.05,
        'formatter.margin.between_channel': 0.1,
        'formatter.label_offset.pulse_name': -0.1,
        'formatter.label_offset.scale_factor': -0.1,
        'formatter.label_offset.frame_change': 0.1,
        'formatter.label_offset.snapshot': 0.1,
        'formatter.text_size.axis_label': 15,
        'formatter.text_size.annotate': 12,
        'formatter.text_size.frame_change': 20,
        'formatter.text_size.snapshot': 20,
        'formatter.text_size.fig_title': 15,
        'formatter.line_width.fill_waveform': 0,
        'formatter.line_width.baseline': 1,
        'formatter.line_width.barrier': 1,
        'formatter.line_style.fill_waveform': '-',
        'formatter.line_style.baseline': '-',
        'formatter.line_style.barrier': ':',
        'formatter.control.apply_phase_modulation': True,
        'formatter.control.show_snapshot_channel': True,
        'formatter.control.show_acquire_channel': True,
        'formatter.control.show_empty_channel': True,
        'formatter.unicode_symbol.frame_change': u'\u21BA',
        'formatter.unicode_symbol.snapshot': u'\u21AF',
        'formatter.latex_symbol.frame_change': r'\circlearrowleft',
        'formatter.latex_symbol.snapshot': '',
        'generator.waveform': [],
        'generator.frame': [],
        'generator.channel': [],
        'generator.snapshot': [],
        'generator.barrier': []}
