# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""latex visualization backends."""

import io
import math
import re
from warnings import warn

import numpy as np
from qiskit.circuit import Clbit, Qubit, ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library.standard_gates import SwapGate, XGate, ZGate, RZZGate, U1Gate, PhaseGate
from qiskit.circuit.measure import Measure
from qiskit.visualization.qcstyle import load_style
from qiskit.circuit.tools.pi_check import pi_check
from .utils import (
    get_gate_ctrl_text,
    get_param_str,
    get_wire_map,
    get_bit_register,
    get_bit_reg_index,
    get_wire_label,
    generate_latex_label,
    get_condition_label_val,
)


class QCircuitImage:
    """This class contains methods to create \\LaTeX circuit images.

    The class targets the \\LaTeX package Q-circuit
    (https://arxiv.org/pdf/quant-ph/0406003).

    Thanks to Eric Sabo for the initial implementation for Qiskit.
    """

    def __init__(
        self,
        qubits,
        clbits,
        nodes,
        scale,
        style=None,
        reverse_bits=False,
        plot_barriers=True,
        layout=None,
        initial_state=False,
        cregbundle=False,
        global_phase=None,
        qregs=None,
        cregs=None,
        with_layout=False,
        circuit=None,
    ):
        """QCircuitImage initializer.

        Args:
            qubits (list[Qubit]): list of qubits
            clbits (list[Clbit]): list of clbits
            nodes (list[list[DAGNode]]): list of circuit instructions, grouped by layer
            scale (float): image scaling
            style (dict or str): dictionary of style or file name of style file
            reverse_bits (bool): when True, reverse the bit ordering of the registers
            plot_barriers (bool): Enable/disable drawing barriers in the output
               circuit. Defaults to True.
            layout (Layout or None): If present, the layout information will be
               included.
            initial_state (bool): Optional. Adds |0> in the beginning of the line. Default: `False`.
            cregbundle (bool): Optional. If set True bundle classical registers. Default: `False`.
            global_phase (float): Optional, the global phase for the circuit.
            circuit (QuantumCircuit): the circuit that's being displayed
        Raises:
            ImportError: If pylatexenc is not installed
        """
        if qregs is not None:
            warn(
                "The 'qregs' kwarg to the QCircuitImage class is deprecated "
                "as of 0.20.0 and will be removed no earlier than 3 months "
                "after the release date.",
                DeprecationWarning,
                2,
            )
        if cregs is not None:
            warn(
                "The 'cregs' kwarg to the QCircuitImage class is deprecated "
                "as of 0.20.0 and will be removed no earlier than 3 months "
                "after the release date.",
                DeprecationWarning,
                2,
            )
        if layout is not None:
            warn(
                "The 'layout' kwarg to the QCircuitImage class is deprecated "
                "as of 0.20.0 and will be removed no earlier than 3 months "
                "after the release date.",
                DeprecationWarning,
                2,
            )
        if global_phase is not None:
            warn(
                "The 'global_phase' kwarg to the QCircuitImage class is deprecated "
                "as of 0.20.0 and will be removed no earlier than 3 months "
                "after the release date.",
                DeprecationWarning,
                2,
            )
        # This check should be removed when the 4 deprecations above are removed
        if circuit is None:
            warn(
                "The 'circuit' kwarg to the QCircuitImage class must be a valid "
                "QuantumCircuit and not None. A new circuit is being created using "
                "the qubits and clbits for rendering the drawing.",
                DeprecationWarning,
                2,
            )
            circ = QuantumCircuit(qubits, clbits)
            for reg in qregs:
                bits = [qubits[circ._qubit_indices[q].index] for q in reg]
                circ.add_register(QuantumRegister(None, reg.name, list(bits)))
            for reg in cregs:
                bits = [clbits[circ._clbit_indices[q].index] for q in reg]
                circ.add_register(ClassicalRegister(None, reg.name, list(bits)))
            self._circuit = circ
        else:
            self._circuit = circuit
        self._qubits = qubits
        self._clbits = clbits

        # list of lists corresponding to layers of the circuit
        self._nodes = nodes

        # image scaling
        self._scale = 1.0 if scale is None else scale

        # Map of cregs to sizes
        self._cregs = {}

        # Array to hold the \\LaTeX commands to generate a circuit image.
        self._latex = []

        # Variable to hold image depth (width)
        self._img_depth = 0

        # Variable to hold image width (height)
        self._img_width = 0

        # Variable to hold total circuit depth
        self._sum_column_widths = 0

        # Variable to hold total circuit width
        self._sum_wire_heights = 0

        # em points of separation between circuit columns
        self._column_separation = 1

        # em points of separation between circuit wire
        self._wire_separation = 0

        # presence of "box" or "target" determines wire spacing
        self._has_box = False
        self._has_target = False

        self._reverse_bits = reverse_bits
        self._plot_barriers = plot_barriers
        if with_layout:
            self._layout = self._circuit._layout
        else:
            self._layout = None

        self._initial_state = initial_state
        self._cregbundle = cregbundle
        self._global_phase = circuit.global_phase

        # If there is any custom instruction that uses classical bits
        # then cregbundle is forced to be False.
        for layer in self._nodes:
            for node in layer:
                if node.op.name not in {"measure"} and node.cargs:
                    self._cregbundle = False

        self._wire_map = get_wire_map(circuit, qubits + clbits, self._cregbundle)
        self._img_width = len(self._wire_map)

        self._style, _ = load_style(style)

    def latex(self):
        """Return LaTeX string representation of circuit."""

        self._initialize_latex_array()
        self._build_latex_array()
        header_1 = r"\documentclass[border=2px]{standalone}" + "\n"

        header_2 = r"""
\usepackage[braket, qm]{qcircuit}
\usepackage{graphicx}

\begin{document}
"""
        header_scale = f"\\scalebox{{{self._scale}}}" + "{"

        qcircuit_line = r"""
\Qcircuit @C=%.1fem @R=%.1fem @!R { \\
"""
        output = io.StringIO()
        output.write(header_1)
        output.write(header_2)
        output.write(header_scale)
        if self._global_phase:
            output.write(
                r"""{$\mathrm{%s} \mathrm{%s}$}"""
                % ("global\\,phase:\\,", pi_check(self._global_phase, output="latex"))
            )
        output.write(qcircuit_line % (self._column_separation, self._wire_separation))
        for i in range(self._img_width):
            output.write("\t \t")
            for j in range(self._img_depth + 1):
                output.write(self._latex[i][j])
                if j != self._img_depth:
                    output.write(" & ")
                else:
                    output.write(r"\\" + "\n")
        output.write(r"\\ " + "}}\n")
        output.write("\\end{document}")
        contents = output.getvalue()
        output.close()
        return contents

    def _initialize_latex_array(self):
        """Initialize qubit and clbit labels and set wire separation"""
        self._img_depth, self._sum_column_widths = self._get_image_depth()
        self._sum_wire_heights = self._img_width
        # choose the most compact wire spacing, while not squashing them
        if self._has_box:
            self._wire_separation = 0.2
        elif self._has_target:
            self._wire_separation = 0.8
        else:
            self._wire_separation = 1.0
        self._latex = [
            ["\\qw" if isinstance(wire, Qubit) else "\\cw" for _ in range(self._img_depth + 1)]
            for wire in self._wire_map
        ]
        self._latex.append([" "] * (self._img_depth + 1))

        # display the bit/register labels
        for wire in self._wire_map:
            if isinstance(wire, ClassicalRegister):
                register = wire
                index = self._wire_map[wire]
            else:
                register, bit_index, reg_index = get_bit_reg_index(
                    self._circuit, wire, self._reverse_bits
                )
                index = bit_index if register is None else reg_index

            wire_label = get_wire_label(
                "latex", register, index, layout=self._layout, cregbundle=self._cregbundle
            )
            wire_label += " : "
            if self._initial_state:
                wire_label += "\\ket{{0}}" if isinstance(wire, Qubit) else "0"
            wire_label += " }"

            if not isinstance(wire, (Qubit)) and self._cregbundle and register is not None:
                pos = self._wire_map[register]
                self._latex[pos][1] = "\\lstick{/_{_{" + str(register.size) + "}}} \\cw"
                wire_label = f"\\mathrm{{{wire_label}}}"
            else:
                pos = self._wire_map[wire]
            self._latex[pos][0] = "\\nghost{" + wire_label + " & " + "\\lstick{" + wire_label

    def _get_image_depth(self):
        """Get depth information for the circuit."""

        # wires in the beginning and end
        columns = 2
        if self._cregbundle and (
            self._nodes
            and self._nodes[0]
            and (self._nodes[0][0].op.name == "measure" or self._nodes[0][0].op.condition)
        ):
            columns += 1

        # Determine wire spacing before image depth
        max_column_widths = []
        for layer in self._nodes:
            column_width = 1
            current_max = 0
            for node in layer:
                op = node.op
                # useful information for determining wire spacing
                boxed_gates = [
                    "u1",
                    "u2",
                    "u3",
                    "u",
                    "p",
                    "x",
                    "y",
                    "z",
                    "h",
                    "s",
                    "sdg",
                    "t",
                    "tdg",
                    "sx",
                    "sxdg",
                    "rx",
                    "ry",
                    "rz",
                    "ch",
                    "cy",
                    "crz",
                    "cu2",
                    "cu3",
                    "cu",
                    "id",
                ]
                target_gates = ["cx", "ccx", "cu1", "cp", "rzz"]
                if op.name in boxed_gates:
                    self._has_box = True
                elif op.name in target_gates:
                    self._has_target = True
                elif isinstance(op, ControlledGate):
                    self._has_box = True

                arg_str_len = 0
                # the wide gates
                for arg in op.params:
                    if not any(isinstance(param, np.ndarray) for param in op.params):
                        arg_str = re.sub(r"[-+]?\d*\.\d{2,}|\d{2,}", self._truncate_float, str(arg))
                        arg_str_len += len(arg_str)

                # the width of the column is the max of all the gates in the column
                current_max = max(arg_str_len, current_max)

                # all gates take up 1 column except from those with side labels (ie cu1, cp, rzz)
                # which take 4 columns
                base_type = None if not hasattr(op, "base_gate") else op.base_gate
                if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
                    column_width = 4
            max_column_widths.append(current_max)
            columns += column_width

        # every 3 characters is roughly one extra 'unit' of width in the cell
        # the gate name is 1 extra 'unit'
        # the qubit/cbit labels plus initial states is 2 more
        # the wires poking out at the ends is 2 more
        sum_column_widths = sum(1 + v / 3 for v in max_column_widths)

        max_wire_name = 3
        for wire in self._wire_map:
            if isinstance(wire, (Qubit, Clbit)):
                register = get_bit_register(self._circuit, wire)
                name = register.name if register is not None else ""
            else:
                name = wire.name
            max_wire_name = max(max_wire_name, len(name))

        sum_column_widths += 5 + max_wire_name / 3

        # could be a fraction so ceil
        return columns, math.ceil(sum_column_widths)

    def _get_beamer_page(self):
        """Get height, width & scale attributes for the beamer page."""

        # PIL python package limits image size to around a quarter gigabyte
        # this means the beamer image should be limited to < 50000
        # if you want to avoid a "warning" too, set it to < 25000
        pil_limit = 40000

        # the beamer latex template limits each dimension to < 19 feet
        # (i.e. 575cm)
        beamer_limit = 550

        # columns are roughly twice as big as wires
        aspect_ratio = self._sum_wire_heights / self._sum_column_widths

        # choose a page margin so circuit is not cropped
        margin_factor = 1.5
        height = min(self._sum_wire_heights * margin_factor, beamer_limit)
        width = min(self._sum_column_widths * margin_factor, beamer_limit)

        # if too large, make it fit
        if height * width > pil_limit:
            height = min(np.sqrt(pil_limit * aspect_ratio), beamer_limit)
            width = min(np.sqrt(pil_limit / aspect_ratio), beamer_limit)

        # if too small, give it a minimum size
        height = max(height, 10)
        width = max(width, 10)

        return (height, width, self._scale)

    def _build_latex_array(self):
        """Returns an array of strings containing \\LaTeX for this circuit."""

        column = 1
        # Leave a column to display number of classical registers if needed
        if self._cregbundle and (
            self._nodes
            and self._nodes[0]
            and (self._nodes[0][0].op.name == "measure" or self._nodes[0][0].op.condition)
        ):
            column += 1

        for layer in self._nodes:
            num_cols_layer = 1

            for node in layer:
                op = node.op
                num_cols_op = 1
                wire_list = [self._wire_map[qarg] for qarg in node.qargs]
                if op.condition:
                    self._add_condition(op, wire_list, column)

                if isinstance(op, Measure):
                    self._build_measure(node, column)

                elif op._directive:  # barrier, snapshot, etc.
                    self._build_barrier(node, column)

                else:
                    gate_text, _, _ = get_gate_ctrl_text(op, "latex", style=self._style)
                    gate_text += get_param_str(op, "latex", ndigits=4)
                    gate_text = generate_latex_label(gate_text)
                    if node.cargs:
                        cwire_list = [self._wire_map[carg] for carg in node.cargs]
                    else:
                        cwire_list = []

                    if len(wire_list) == 1 and not node.cargs:
                        self._latex[wire_list[0]][column] = "\\gate{%s}" % gate_text

                    elif isinstance(op, ControlledGate):
                        num_cols_op = self._build_ctrl_gate(op, gate_text, wire_list, column)
                    else:
                        num_cols_op = self._build_multi_gate(
                            op, gate_text, wire_list, cwire_list, column
                        )

                num_cols_layer = max(num_cols_layer, num_cols_op)

            column += num_cols_layer

    def _build_multi_gate(self, op, gate_text, wire_list, cwire_list, col):
        """Add a multiple wire gate to the _latex list"""
        cwire_start = len(self._qubits)
        num_cols_op = 1
        if isinstance(op, (SwapGate, RZZGate)):
            num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
        else:
            wire_min = min(wire_list)
            wire_max = max(wire_list)
            if cwire_list and not self._cregbundle:
                wire_max = max(cwire_list)
            wire_ind = wire_list.index(wire_min)
            self._latex[wire_min][col] = (
                f"\\multigate{{{wire_max - wire_min}}}{{{gate_text}}}_"
                + "<" * (len(str(wire_ind)) + 2)
                + "{%s}" % wire_ind
            )
            for wire in range(wire_min + 1, wire_max + 1):
                if wire < cwire_start:
                    ghost_box = "\\ghost{%s}" % gate_text
                    if wire in wire_list:
                        wire_ind = wire_list.index(wire)
                else:
                    ghost_box = "\\cghost{%s}" % gate_text
                    if wire in cwire_list:
                        wire_ind = cwire_list.index(wire)
                if wire in wire_list + cwire_list:
                    self._latex[wire][col] = (
                        ghost_box + "_" + "<" * (len(str(wire_ind)) + 2) + "{%s}" % wire_ind
                    )
                else:
                    self._latex[wire][col] = ghost_box
        return num_cols_op

    def _build_ctrl_gate(self, op, gate_text, wire_list, col):
        """Add a gate with multiple controls to the _latex list"""
        num_cols_op = 1
        num_ctrl_qubits = op.num_ctrl_qubits
        wireqargs = wire_list[num_ctrl_qubits:]
        ctrlqargs = wire_list[:num_ctrl_qubits]
        wire_min = min(wireqargs)
        wire_max = max(wireqargs)
        ctrl_state = f"{op.ctrl_state:b}".rjust(num_ctrl_qubits, "0")[::-1]

        # First do single qubit target gates
        if len(wireqargs) == 1:
            self._add_controls(wire_list, ctrlqargs, ctrl_state, col)

            # Check for cx, cz, cu1 and cp first, then do standard gate
            if isinstance(op.base_gate, XGate):
                self._latex[wireqargs[0]][col] = "\\targ"
            elif isinstance(op.base_gate, ZGate):
                self._latex[wireqargs[0]][col] = "\\control\\qw"
            elif isinstance(op.base_gate, (U1Gate, PhaseGate)):
                num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
            else:
                self._latex[wireqargs[0]][col] = "\\gate{%s}" % gate_text
        else:
            # Treat special cases of swap and rzz gates
            if isinstance(op.base_gate, (SwapGate, RZZGate)):
                self._add_controls(wire_list, ctrlqargs, ctrl_state, col)
                num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
            else:
                # If any controls appear in the span of the multiqubit
                # gate just treat the whole thing as a big gate
                for ctrl in ctrlqargs:
                    if ctrl in range(wire_min, wire_max):
                        wireqargs = wire_list
                        break
                else:
                    self._add_controls(wire_list, ctrlqargs, ctrl_state, col)

                self._build_multi_gate(op, gate_text, wireqargs, [], col)
        return num_cols_op

    def _build_symmetric_gate(self, op, gate_text, wire_list, col):
        """Add symmetric gates for cu1, cp, swap, and rzz"""
        wire_max = max(wire_list)
        # The last and next to last in the wire list are the gate wires without added controls
        wire_next_last = wire_list[-2]
        wire_last = wire_list[-1]
        base_op = None if not hasattr(op, "base_gate") else op.base_gate

        if isinstance(op, SwapGate) or (base_op and isinstance(base_op, SwapGate)):
            self._latex[wire_next_last][col] = "\\qswap"
            self._latex[wire_last][col] = "\\qswap \\qwx[" + str(wire_next_last - wire_last) + "]"
            return 1  # num_cols

        if isinstance(op, RZZGate) or (base_op and isinstance(base_op, RZZGate)):
            ctrl_bit = "1"
        else:
            ctrl_bit = f"{op.ctrl_state:b}".rjust(1, "0")[::-1]

        control = "\\ctrlo" if ctrl_bit == "0" else "\\ctrl"
        self._latex[wire_next_last][col] = f"{control}" + (
            "{" + str(wire_last - wire_next_last) + "}"
        )
        self._latex[wire_last][col] = "\\control \\qw"
        # Put side text to the right between bottom wire in wire_list and the one above it
        self._latex[wire_max - 1][col + 1] = "\\dstick{\\hspace{2.0em}%s} \\qw" % gate_text
        return 4  # num_cols for side text gates

    def _build_measure(self, node, col):
        """Build a meter and the lines to the creg"""
        wire1 = self._wire_map[node.qargs[0]]
        self._latex[wire1][col] = "\\meter"

        idx_str = ""
        cond_offset = 1.5 if node.op.condition else 0.0
        if self._cregbundle:
            register = get_bit_register(self._circuit, node.cargs[0])
            if register is not None:
                wire2 = self._wire_map[register]
                idx_str = str(node.cargs[0].index)
            else:
                wire2 = self._wire_map[node.cargs[0]]

            self._latex[wire2][col] = "\\dstick{_{_{\\hspace{%sem}%s}}} \\cw \\ar @{<=} [-%s,0]" % (
                cond_offset,
                idx_str,
                str(wire2 - wire1),
            )
        else:
            wire2 = self._wire_map[node.cargs[0]]
            self._latex[wire2][col] = "\\cw \\ar @{<=} [-" + str(wire2 - wire1) + ",0]"

    def _build_barrier(self, node, col):
        """Build a partial or full barrier if plot_barriers set"""
        if self._plot_barriers:
            indexes = [self._wire_map[qarg] for qarg in node.qargs]
            indexes.sort()
            first = last = indexes[0]
            for index in indexes[1:]:
                if index - 1 == last:
                    last = index
                else:
                    pos = self._wire_map[self._qubits[first]]
                    self._latex[pos][col - 1] += " \\barrier[0em]{" + str(last - first) + "}"
                    self._latex[pos][col] = "\\qw"
                    first = last = index
            pos = self._wire_map[self._qubits[first]]
            self._latex[pos][col - 1] += " \\barrier[0em]{" + str(last - first) + "}"
            self._latex[pos][col] = "\\qw"

    def _add_controls(self, wire_list, ctrlqargs, ctrl_state, col):
        """Add one or more controls to a gate"""
        for index, ctrl_item in enumerate(zip(ctrlqargs, ctrl_state)):
            pos = ctrl_item[0]
            nxt = wire_list[index]
            if wire_list[index] > wire_list[-1]:
                nxt -= 1
                while nxt not in wire_list:
                    nxt -= 1
            else:
                nxt += 1
                while nxt not in wire_list:
                    nxt += 1

            # ctrl_item[1] is ctrl_state for this bit
            control = "\\ctrlo" if ctrl_item[1] == "0" else "\\ctrl"
            self._latex[pos][col] = f"{control}" + "{" + str(nxt - wire_list[index]) + "}"

    def _add_condition(self, op, wire_list, col):
        """Add a condition to the _latex list"""
        # cwire - the wire number for the first wire for the condition register
        #         or if cregbundle, wire number of the condition register itself
        # gap - the number of wires from cwire to the bottom gate qubit

        label, val_bits = get_condition_label_val(
            op.condition, self._circuit, self._cregbundle, self._reverse_bits
        )
        cond_is_bit = isinstance(op.condition[0], Clbit)
        cond_reg = op.condition[0]
        if cond_is_bit:
            register = get_bit_register(self._circuit, op.condition[0])
            if register is not None:
                cond_reg = register

        if self._cregbundle:
            cwire = self._wire_map[cond_reg]
        else:
            cwire = self._wire_map[op.condition[0] if cond_is_bit else cond_reg[0]]

        gap = cwire - max(wire_list)
        meas_offset = -0.3 if isinstance(op, Measure) else 0.0

        # Print the condition value at the bottom and put bullet on creg line
        if cond_is_bit or self._cregbundle:
            control = "\\control" if op.condition[1] else "\\controlo"
            self._latex[cwire][col] = f"{control}" + " \\cw^(%s){^{\\mathtt{%s}}} \\cwx[-%s]" % (
                meas_offset,
                label,
                str(gap),
            )
        else:
            cond_len = op.condition[0].size - 1
            # If reverse, start at highest reg bit and go down to 0
            if self._reverse_bits:
                cwire -= cond_len
                gap -= cond_len
            # Iterate through the reg bits down to the lowest one
            for i in range(cond_len):
                control = "\\control" if val_bits[i] == "1" else "\\controlo"
                self._latex[cwire + i][col] = f"{control} \\cw \\cwx[-" + str(gap) + "]"
                gap = 1
            # Add (hex condition value) below the last cwire
            control = "\\control" if val_bits[cond_len] == "1" else "\\controlo"
            self._latex[cwire + cond_len][col] = (
                f"{control}" + " \\cw^(%s){^{\\mathtt{%s}}} \\cwx[-%s]"
            ) % (
                meas_offset,
                label,
                str(gap),
            )

    def _truncate_float(self, matchobj, ndigits=4):
        """Truncate long floats."""
        if matchobj.group(0):
            return f"%.{ndigits}g" % float(matchobj.group(0))
        return ""
