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

The `QiskitPulseStyle` is a general stylesheet template.
The actual stylesheet instance is initialized with the hard corded default values
when the pulse drawer module is loaded so that every lower modules can access to the information.

The `QiskitPulseStyle` takes nested python dictionary and stores the settings under the
flattened dictionary key such as `formatter.color.background`.
This key representation and initialization are the imitative of `rcParams` of `matplotlib`.
However, the `QiskitPulseStyle` is generally not compatible with the `rcParams`
because the pulse stylesheet is heavily specialized to the context of the pulse program.


Stylesheets
~~~~~~~~~~~
The standard stylesheets provided by the pulse drawer are the subclass of the python dictionary.
Thus, the stylesheets can provide sufficient information about their preferences in the docstring.
However an end-user can still create a custom stylesheet just as a nested dictionary:

    ```python
    my_favorite_style = {
        'formatter': {
            'color': {'background': '#ffffff'}
        }
    }
    ```

Note that the user just need to write necessary settings to update.
Those stylesheets are fed into the drawer interface and the output images are modified
according to the provided preferences.

The settings of stylesheet are broadly separated into `formatter` and `generator`.

Formatter
~~~~~~~~~
The `formatter` is a dictionary of parameters to control the appearance of visualization elements.
This is very close data structure to the `rcParams` of `matplotlib`.
However, the user can add custom keys that are used by the custom generators.

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
- `formatter.color.symbol`: Color code for symbols.
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
- formatter.layer.symbol`: Layer position of symbols.

Margins (float):
- `formatter.margin.top`: Top margin of the canvas in units of pulse height.
- `formatter.margin.bottom`: Bottom margin of the canvas in units of pulse height.
- `formatter.margin.left`: Left margin of the canvas in units of schedule length.
- `formatter.margin.right`: Right margin of the canvas in units of schedule length.
- `formatter.margin.between_channel`: Spacing between channels in units of pulse height.

Text size (float)
- `formatter.text_size.axis_label`: Text size of axis labels.
- `formatter.text_size.annotate`: Text size of annotations.
- `formatter.text_size.symbol`: Text size of symbols.
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
User can add custom functions to draw new drawing objects.

- `generator.waveform`: Generate drawing objects from waveform type instructions.
- `generator.frame`: Generate drawing objects from frame type instructions.
- `generator.channel`: Generate drawing objects from channel information.
- `generator.snapshot`: Generate drawing objects from snapshot information.
- `generator.barrier`: Generate drawing objects from barrier information.


Contributors should write a description of settings here when new default values are added.
"""

from typing import Dict, Any

from qiskit.visualization.pulse_v2 import generators, ComplexColors


class QiskitPulseStyle:
    """Stylesheet for pulse drawer.
    """
    def __init__(self):
        self._style = dict()

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, new_style):
        flat_dict = _flatten_dict(new_style)
        self._style.update(flat_dict)


def _flatten_dict(nested_dict: Dict[str, Any],
                  parent: str = None):
    items = []
    for key, val in nested_dict.items():
        concatenated_key = '{}.{}'.format(parent, key) if parent else key
        if isinstance(val, dict):
            items.extend(_flatten_dict(val, parent=concatenated_key).items())
        else:
            items.append((concatenated_key, val))

    return dict(items)


class IqxStandard(dict):
    """Standard pulse stylesheet.

    - Generate stepwise waveform envelope with latex pulse names.
    - Plot frame change symbol with raw operand values.
    - Show latex channel name with channel's scaling factor.
    - Show snapshot and barrier.
    """
    def __init__(self):
        super().__init__()
        style = {
            'formatter': {
                'control': {
                    'apply_phase_modulation': True,
                    'show_snapshot_channel': True,
                    'show_acquire_channel': True,
                    'show_empty_channel': False,
                }
            },
            'generator': {
                'waveform': [generators.gen_filled_waveform_stepwise,
                             generators.gen_iqx_latex_waveform_name],
                'frame': [generators.gen_frame_symbol,
                          generators.gen_raw_frame_operand_values],
                'channel': [generators.gen_latex_channel_name,
                            generators.gen_scaling_info,
                            generators.gen_baseline],
                'snapshot': [generators.gen_snapshot_symbol],
                'barrier': [generators.gen_barrier]
            }
        }
        self.update(style)

    def __repr__(self):
        return 'Standard Pulse style sheet.'


class IqxPublication(dict):
    """Simple pulse stylesheet suited for publication.

    - Generate stepwise waveform envelope with latex pulse names.
    - Plot frame change symbol with latex VZ notation. No frequency.
    - Show latex channel name with channel's scaling factor.
    - Do not show snapshot and barrier.
    """

    def __init__(self):
        super().__init__()
        style = {
            'formatter': {
                'control': {
                    'apply_phase_modulation': True,
                    'show_snapshot_channel': False,
                    'show_acquire_channel': False,
                    'show_empty_channel': False,
                }
            },
            'generator': {
                'waveform': [generators.gen_filled_waveform_stepwise,
                             generators.gen_iqx_latex_waveform_name],
                'frame': [generators.gen_frame_symbol,
                          generators.gen_latex_vz_label],
                'channel': [generators.gen_latex_channel_name,
                            generators.gen_scaling_info,
                            generators.gen_baseline],
                'snapshot': [],
                'barrier': []
            }
        }

        self.update(style)

    def __repr__(self):
        return 'Simple pulse style sheet for publication.'


class IqxDebugging(dict):
    """Pulse stylesheet for pulse programmers. Show details of instructions.

    # TODO: add more generators

    - Generate stepwise waveform envelope with latex pulse names.
    - Plot frame change symbol with raw operand values.
    - Show latex channel name with channel's scaling factor.
    - Show snapshot and barrier.
    """

    def __init__(self):
        super().__init__()
        style = {
            'formatter': {
                'control': {
                    'apply_phase_modulation': True,
                    'show_snapshot_channel': True,
                    'show_acquire_channel': True,
                    'show_empty_channel': True,
                }
            },
            'generator': {
                'waveform': [generators.gen_filled_waveform_stepwise,
                             generators.gen_iqx_latex_waveform_name],
                'frame': [generators.gen_frame_symbol,
                          generators.gen_raw_frame_operand_values],
                'channel': [generators.gen_latex_channel_name,
                            generators.gen_scaling_info,
                            generators.gen_baseline],
                'snapshot': [generators.gen_snapshot_symbol],
                'barrier': [generators.gen_barrier]
            }
        }

        self.update(style)

    def __repr__(self):
        return 'Pulse style sheet for pulse programmers.'


hard_corded_default_style = {
    'formatter': {
        'general': {
            'fig_size': (8, 6),
            'dpi': 150
        },
        'color': {
            'fill_waveform_d': ComplexColors('#648fff', '#002999'),
            'fill_waveform_u': ComplexColors('#ffb000', '#994A00'),
            'fill_waveform_m': ComplexColors('#dc267f', '#760019'),
            'fill_waveform_a': ComplexColors('#dc267f', '#760019'),
            'baseline': '#000000',
            'barrier': '#222222',
            'background': '#f2f3f4',
            'annotate': '#222222',
            'symbol': '#000000',
            'axis_label': '#000000'
        },
        'alpha': {
            'fill_waveform': 1.0,
            'baseline': 1.0,
            'barrier': 0.7
        },
        'layer': {
            'fill_waveform': 2,
            'baseline': 1,
            'barrier': 1,
            'annotate': 4,
            'axis_label': 4,
            'symbol': 3
        },
        'margin': {
            'top': 0.2,
            'bottom': 0.2,
            'left': 0.05,
            'right': 0.05,
            'between_channel': 0.1
        },
        'text_size': {
            'axis_label': 15,
            'annotate': 12,
            'symbol': 20,
            'fig_title': 15
        },
        'line_width': {
            'fill_waveform': 0,
            'baseline': 1,
            'barrier': 1
        },
        'line_style': {
            'fill_waveform': '-',
            'baseline': '-',
            'barrier': ':'
        },
        'control': {
            'apply_phase_modulation': True,
            'show_snapshot_channel': False,
            'show_acquire_channel': False,
            'show_empty_channel': False
        },
        'unicode_symbol': {
            'frame_change': u'\u21BA',
            'snapshot': u'\u21AF'
        },
        'latex_symbol': {
            'frame_change': r'\circlearrowleft',
            'snapshot': None
        }
    },
    'generator': {
        'waveform': None,
        'frame': None,
        'channel': None,
        'snapshot': None,
        'barrier': None
    }
}
