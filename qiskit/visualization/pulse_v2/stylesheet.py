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

The settings of stylesheet are broadly separated into `formatter`, `generator` and `layout`.
The formatter is a nested dictionary of drawing parameters to control the appearance of
each visualization element. This takes similar data structure to the `rcParams` of `matplotlib`.
The generator is a list of callback functions that generates drawing objects from
given program and device data. The layout is a callback function that determines
the appearance of the output image.
"""

from typing import Dict, Any, Mapping
from qiskit.visualization.pulse_v2 import generators, layouts


class QiskitPulseStyle(dict):
    """Stylesheet for pulse drawer."""

    def __init__(self):
        super().__init__()
        # to inform which stylesheet is applied. some plotter may not support specific style.
        self.stylesheet = None
        self.update(default_style())

    def update(self, __m: Mapping[str, Any], **kwargs) -> None:
        super().update(__m, **kwargs)
        for key, value in __m.items():
            self.__setitem__(key, value)
        self.stylesheet = __m.__class__.__name__

    @property
    def formatter(self):
        """Return formatter field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split(".")
            if sub_keys[0] == "formatter":
                sub_dict[".".join(sub_keys[1:])] = value
        return sub_dict

    @property
    def generator(self):
        """Return generator field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split(".")
            if sub_keys[0] == "generator":
                sub_dict[".".join(sub_keys[1:])] = value
        return sub_dict

    @property
    def layout(self):
        """Return layout field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split(".")
            if sub_keys[0] == "layout":
                sub_dict[".".join(sub_keys[1:])] = value
        return sub_dict


class IQXStandard(dict):
    """Standard pulse stylesheet.

    - Generate stepwise waveform envelope with latex pulse names.
    - Apply phase modulation to waveforms.
    - Plot frame change symbol with formatted operand values.
    - Show chart name with scaling factor.
    - Show snapshot and barrier.
    - Do not show acquire channels.
    - Channels are sorted by index and control channels are added to the end.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {
            "formatter.control.apply_phase_modulation": True,
            "formatter.control.show_snapshot_channel": True,
            "formatter.control.show_acquire_channel": False,
            "formatter.control.show_empty_channel": False,
            "formatter.control.auto_chart_scaling": True,
            "formatter.control.axis_break": True,
            "generator.waveform": [
                generators.gen_filled_waveform_stepwise,
                generators.gen_ibmq_latex_waveform_name,
            ],
            "generator.frame": [generators.gen_frame_symbol, generators.gen_formatted_frame_values],
            "generator.chart": [
                generators.gen_chart_name,
                generators.gen_baseline,
                generators.gen_channel_freqs,
            ],
            "generator.snapshot": [generators.gen_snapshot_symbol],
            "generator.barrier": [generators.gen_barrier],
            "layout.chart_channel_map": layouts.channel_index_grouped_sort_u,
            "layout.time_axis_map": layouts.time_map_in_ns,
            "layout.figure_title": layouts.detail_title,
        }
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return "Standard Pulse style sheet."


class IQXSimple(dict):
    """Simple pulse stylesheet without channel notation.

    - Generate stepwise waveform envelope with latex pulse names.
    - Apply phase modulation to waveforms.
    - Do not show frame changes.
    - Show chart name.
    - Do not show snapshot and barrier.
    - Do not show acquire channels.
    - Channels are sorted by qubit index.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {
            "formatter.general.fig_chart_height": 5,
            "formatter.control.apply_phase_modulation": True,
            "formatter.control.show_snapshot_channel": True,
            "formatter.control.show_acquire_channel": False,
            "formatter.control.show_empty_channel": False,
            "formatter.control.auto_chart_scaling": False,
            "formatter.control.axis_break": True,
            "generator.waveform": [
                generators.gen_filled_waveform_stepwise,
                generators.gen_ibmq_latex_waveform_name,
            ],
            "generator.frame": [],
            "generator.chart": [generators.gen_chart_name, generators.gen_baseline],
            "generator.snapshot": [],
            "generator.barrier": [],
            "layout.chart_channel_map": layouts.qubit_index_sort,
            "layout.time_axis_map": layouts.time_map_in_ns,
            "layout.figure_title": layouts.empty_title,
        }
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return "Simple pulse style sheet for publication."


class IQXDebugging(dict):
    """Pulse stylesheet for pulse programmers. Show details of instructions.

    # TODO: add more generators

    - Generate stepwise waveform envelope with latex pulse names.
    - Generate annotation for waveform height.
    - Apply phase modulation to waveforms.
    - Plot frame change symbol with raw operand values.
    - Show chart name and channel frequency.
    - Show snapshot and barrier.
    - Show acquire channels.
    - Channels are sorted by index and control channels are added to the end.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {
            "formatter.control.apply_phase_modulation": True,
            "formatter.control.show_snapshot_channel": True,
            "formatter.control.show_acquire_channel": True,
            "formatter.control.show_empty_channel": False,
            "formatter.control.auto_chart_scaling": True,
            "formatter.control.axis_break": True,
            "generator.waveform": [
                generators.gen_filled_waveform_stepwise,
                generators.gen_ibmq_latex_waveform_name,
                generators.gen_waveform_max_value,
            ],
            "generator.frame": [
                generators.gen_frame_symbol,
                generators.gen_raw_operand_values_compact,
            ],
            "generator.chart": [
                generators.gen_chart_name,
                generators.gen_baseline,
                generators.gen_channel_freqs,
            ],
            "generator.snapshot": [generators.gen_snapshot_symbol, generators.gen_snapshot_name],
            "generator.barrier": [generators.gen_barrier],
            "layout.chart_channel_map": layouts.channel_index_grouped_sort_u,
            "layout.time_axis_map": layouts.time_map_in_ns,
            "layout.figure_title": layouts.detail_title,
        }
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return "Pulse style sheet for pulse programmers."


def default_style() -> Dict[str, Any]:
    """Define default values of the pulse stylesheet."""
    return {
        "formatter.general.fig_width": 13,
        "formatter.general.fig_chart_height": 1.5,
        "formatter.general.vertical_resolution": 1e-6,
        "formatter.general.max_scale": 100,
        "formatter.color.waveforms": {
            "W": ["#648fff", "#002999"],
            "D": ["#648fff", "#002999"],
            "U": ["#ffb000", "#994A00"],
            "M": ["#dc267f", "#760019"],
            "A": ["#dc267f", "#760019"],
        },
        "formatter.color.baseline": "#000000",
        "formatter.color.barrier": "#222222",
        "formatter.color.background": "#f2f3f4",
        "formatter.color.fig_title": "#000000",
        "formatter.color.annotate": "#222222",
        "formatter.color.frame_change": "#000000",
        "formatter.color.snapshot": "#000000",
        "formatter.color.axis_label": "#000000",
        "formatter.color.opaque_shape": ["#f2f3f4", "#000000"],
        "formatter.alpha.fill_waveform": 0.3,
        "formatter.alpha.baseline": 1.0,
        "formatter.alpha.barrier": 0.7,
        "formatter.alpha.opaque_shape": 0.7,
        "formatter.layer.fill_waveform": 2,
        "formatter.layer.baseline": 1,
        "formatter.layer.barrier": 1,
        "formatter.layer.annotate": 5,
        "formatter.layer.axis_label": 5,
        "formatter.layer.frame_change": 4,
        "formatter.layer.snapshot": 3,
        "formatter.layer.fig_title": 6,
        "formatter.margin.top": 0.5,
        "formatter.margin.bottom": 0.5,
        "formatter.margin.left_percent": 0.05,
        "formatter.margin.right_percent": 0.05,
        "formatter.margin.between_channel": 0.5,
        "formatter.label_offset.pulse_name": 0.3,
        "formatter.label_offset.chart_info": 0.3,
        "formatter.label_offset.frame_change": 0.3,
        "formatter.label_offset.snapshot": 0.3,
        "formatter.text_size.axis_label": 15,
        "formatter.text_size.annotate": 12,
        "formatter.text_size.frame_change": 20,
        "formatter.text_size.snapshot": 20,
        "formatter.text_size.fig_title": 15,
        "formatter.text_size.axis_break_symbol": 15,
        "formatter.line_width.fill_waveform": 0,
        "formatter.line_width.axis_break": 6,
        "formatter.line_width.baseline": 1,
        "formatter.line_width.barrier": 1,
        "formatter.line_width.opaque_shape": 1,
        "formatter.line_style.fill_waveform": "-",
        "formatter.line_style.baseline": "-",
        "formatter.line_style.barrier": ":",
        "formatter.line_style.opaque_shape": "--",
        "formatter.channel_scaling.drive": 1.0,
        "formatter.channel_scaling.control": 1.0,
        "formatter.channel_scaling.measure": 1.0,
        "formatter.channel_scaling.acquire": 1.0,
        "formatter.channel_scaling.pos_spacing": 0.1,
        "formatter.channel_scaling.neg_spacing": -0.1,
        "formatter.box_width.opaque_shape": 150,
        "formatter.box_height.opaque_shape": 0.5,
        "formatter.axis_break.length": 3000,
        "formatter.axis_break.max_length": 1000,
        "formatter.control.fill_waveform": True,
        "formatter.control.apply_phase_modulation": True,
        "formatter.control.show_snapshot_channel": True,
        "formatter.control.show_acquire_channel": True,
        "formatter.control.show_empty_channel": True,
        "formatter.control.auto_chart_scaling": True,
        "formatter.control.axis_break": True,
        "formatter.unicode_symbol.frame_change": "\u21BA",
        "formatter.unicode_symbol.snapshot": "\u21AF",
        "formatter.unicode_symbol.phase_parameter": "\u03b8",
        "formatter.unicode_symbol.freq_parameter": "f",
        "formatter.latex_symbol.frame_change": r"\circlearrowleft",
        "formatter.latex_symbol.snapshot": "",
        "formatter.latex_symbol.phase_parameter": r"\theta",
        "formatter.latex_symbol.freq_parameter": "f",
        "generator.waveform": [],
        "generator.frame": [],
        "generator.chart": [],
        "generator.snapshot": [],
        "generator.barrier": [],
        "layout.chart_channel_map": None,
        "layout.time_axis_map": None,
        "layout.figure_title": None,
    }
