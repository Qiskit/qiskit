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

The general stylesheet template `QiskitPulseStyle` is initialized with the hard-corded
default values in the `default.json` file. This instance is generated when the pulse drawer
module is loaded so that every lower modules can access to the information.

The `QiskitPulseStyle` takes nested python dictionary and stores the settings under
flattened dictionary keys such as `formatter.color.background`.
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

import json
import os
from typing import Dict, Any
import warnings


class QiskitPulseStyle:
    """Stylesheet for pulse drawer.
    """
    def __init__(self):
        self._style = dict()

    @property
    def style(self):
        """Return style dictionary."""
        return self._style

    @style.setter
    def style(self, new_style):
        """Set nested style dictionary."""
        flat_dict = _flatten_dict(new_style)

        current_style = dict()
        for key, val in flat_dict.items():
            current_style[_replace_deprecated_key(key)] = val

        self._style.update(flat_dict)


def _flatten_dict(nested_dict: Dict[str, Any],
                  parent: str = None) -> Dict[str, Any]:
    """A helper function to flatten the nested dictionary.

    Args:
        nested_dict: A nested python dictionary.
        parent: A key of items in the parent dictionary.

    Returns:
        Flattened dictionary.
    """
    items = []
    for key, val in nested_dict.items():
        concatenated_key = '{}.{}'.format(parent, key) if parent else key
        if isinstance(val, dict):
            items.extend(_flatten_dict(val, parent=concatenated_key).items())
        else:
            items.append((concatenated_key, val))

    return dict(items)


def _replace_deprecated_key(key: str) -> str:
    """A helper function to replace deprecated key.

    Args:
        key: Key to check.

    Returns:
        Key in the latest version.
    """
    _replace_table = {}

    if key in _replace_table:
        warnings.warn('%s is deprecated. Use %s instead.' % (key, _replace_table[key]),
                      DeprecationWarning)

        return _replace_table[key]

    return key


def init_style_from_file() -> QiskitPulseStyle:
    """Initialize stylesheet with default setting file."""
    default_style = QiskitPulseStyle()

    dirname = os.path.dirname(__file__)
    filename = "default.json"
    with open(os.path.join(dirname, filename), "r") as f_default:
        default_dict = json.load(f_default)

    default_style.style = default_dict
    return default_style
