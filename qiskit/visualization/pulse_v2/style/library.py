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
Stylesheet library of the pulse drawer.

The standard stylesheets provided by the pulse drawer are the subclass of the python dictionary.
This enables a stylesheet to provide sufficient information of preferences in the docstring.
An end-user can still create a custom stylesheet as a python dictionary:

    ```python
    my_favorite_style = {'formatter.color.background': '#ffffff'}
    ```

Note that the user can partially update a default stylesheet with py:method:`dict.update`.

Those stylesheets are fed into the drawer interface and the output images are modified
according to the provided preferences.
"""

from qiskit.visualization.pulse_v2 import generators


class IqxStandard(dict):
    """Standard pulse stylesheet.

    - Generate stepwise waveform envelope with latex pulse names.
    - Plot frame change symbol with raw operand values.
    - Show latex channel name with channel's scaling factor.
    - Show snapshot and barrier.
    """
    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.apply_phase_modulation': True,
                 'formatter.control.show_snapshot_channel': True,
                 'formatter.control.show_acquire_channel': True,
                 'formatter.control.show_empty_channel': False,
                 'generator.waveform': [generators.gen_filled_waveform_stepwise,
                                        generators.gen_iqx_latex_waveform_name],
                 'generator.frame': [generators.gen_frame_symbol,
                                     generators.gen_raw_frame_operand_values],
                 'generator.channel': [generators.gen_latex_channel_name,
                                       generators.gen_scaling_info,
                                       generators.gen_baseline],
                 'generator.snapshot': [generators.gen_snapshot_symbol],
                 'generator.barrier': [generators.gen_barrier]}
        style.update(**kwargs)
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

    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.apply_phase_modulation': True,
                 'formatter.control.show_snapshot_channel': False,
                 'formatter.control.show_acquire_channel': False,
                 'formatter.control.show_empty_channel': False,
                 'generator.waveform': [generators.gen_filled_waveform_stepwise,
                                        generators.gen_iqx_latex_waveform_name],
                 'generator.frame': [generators.gen_frame_symbol,
                                     generators.gen_latex_vz_label],
                 'generator.channel': [generators.gen_latex_channel_name,
                                       generators.gen_scaling_info,
                                       generators.gen_baseline],
                 'generator.snapshot': [],
                 'generator.barrier': []}
        style.update(**kwargs)
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

    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.apply_phase_modulation': True,
                 'formatter.control.show_snapshot_channel': True,
                 'formatter.control.show_acquire_channel': True,
                 'formatter.control.show_empty_channel': True,
                 'generator.waveform': [generators.gen_filled_waveform_stepwise,
                                        generators.gen_iqx_latex_waveform_name],
                 'generator.frame': [generators.gen_frame_symbol,
                                     generators.gen_raw_frame_operand_values],
                 'generator.channel': [generators.gen_latex_channel_name,
                                       generators.gen_scaling_info,
                                       generators.gen_baseline],
                 'generator.snapshot': [generators.gen_snapshot_symbol],
                 'generator.barrier': [generators.gen_barrier]}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Pulse style sheet for pulse programmers.'
