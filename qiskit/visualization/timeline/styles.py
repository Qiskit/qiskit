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

from qiskit.visualization.timeline import generators, layouts


class IqxStandard(dict):
    """Standard timeline stylesheet.

    - Show time buckets.
    - Show only operand name.
    - Show bit name.
    - Show bit link.
    - Remove idle bits.
    - Remove classical bits.
    """
    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.show_idle': False,
                 'formatter.control.show_clbits': False,
                 'formatter.control.show_barriers': False,
                 'formatter.control.show_delays': False,
                 'generator.gates': [generators.gen_sched_gate,
                                     generators.gen_short_gate_name],
                 'generator.bits': [generators.gen_bit_name,
                                    generators.gen_timeslot],
                 'generator.barriers': [generators.gen_barrier],
                 'generator.bit_links': [generators.gen_bit_link],
                 'layout.gate_color': layouts.default_color_table,
                 'layout.latex_gate_name': layouts.default_latex_gate_name,
                 'layout.bit_arrange': layouts.qreg_creg_ascending}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Standard timeline style sheet.'


class IqxSimple(dict):
    """Simple timeline stylesheet.

    - Show time buckets.
    - Show bit name.
    - Show bit link.
    - Remove idle bits.
    - Remove classical bits.
    """
    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.show_idle': False,
                 'formatter.control.show_clbits': False,
                 'formatter.control.show_barriers': False,
                 'formatter.control.show_delays': False,
                 'generator.gates': [generators.gen_sched_gate],
                 'generator.bits': [generators.gen_bit_name,
                                    generators.gen_timeslot],
                 'generator.barriers': [generators.gen_barrier],
                 'generator.bit_links': [generators.gen_bit_link],
                 'layout.gate_color': layouts.default_color_table,
                 'layout.latex_gate_name': layouts.default_latex_gate_name,
                 'layout.bit_arrange': layouts.qreg_creg_ascending}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Simplified timeline style sheet.'


class IqxDebugging(dict):
    """Timeline stylesheet for programmers. Show details of instructions.

    - Show time buckets.
    - Show operand name, qubits, and parameters.
    - Show barriers.
    - Show delays.
    - Show bit name.
    - Show bit link.
    """
    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.show_idle': True,
                 'formatter.control.show_clbits': True,
                 'formatter.control.show_barriers': True,
                 'formatter.control.show_delays': True,
                 'generator.gates': [generators.gen_sched_gate,
                                     generators.gen_full_gate_name],
                 'generator.bits': [generators.gen_bit_name,
                                    generators.gen_timeslot],
                 'generator.barriers': [generators.gen_barrier],
                 'generator.bit_links': [generators.gen_bit_link],
                 'layout.gate_color': layouts.default_color_table,
                 'layout.latex_gate_name': layouts.default_latex_gate_name,
                 'layout.bit_arrange': layouts.qreg_creg_ascending}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Timeline style sheet for timeline programmers.'
