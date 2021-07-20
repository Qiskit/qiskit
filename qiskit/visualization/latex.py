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

import numpy as np
from qiskit.circuit import Clbit
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library.standard_gates import SwapGate, XGate, ZGate, RZZGate, U1Gate, PhaseGate
from qiskit.circuit.measure import Measure
from qiskit.visualization.qcstyle import load_style
from qiskit.visualization import exceptions
from qiskit.circuit.tools.pi_check import pi_check
from .utils import get_gate_ctrl_text, get_param_str, generate_latex_label


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
            qregs (list): List qregs present in the circuit.
            cregs (list): List of cregs present in the circuit.
        Raises:
            ImportError: If pylatexenc is not installed
        """
        # list of lists corresponding to layers of the circuit
        self.nodes = nodes

        # image scaling
        self.scale = 1.0 if scale is None else scale

        # Map of qregs to sizes
        self.qregs = {}

        # Map of cregs to sizes
        self.cregs = {}

        # List of qubits and cbits in order of appearance in code and image
        # May also include ClassicalRegisters if cregbundle=True
        self.ordered_bits = []

        # Map from registers to the list they appear in the image
        self.img_regs = {}

        # Array to hold the \\LaTeX commands to generate a circuit image.
        self._latex = []

        # Variable to hold image depth (width)
        self.img_depth = 0

        # Variable to hold image width (height)
        self.img_width = 0

        # Variable to hold total circuit depth
        self.sum_column_widths = 0

        # Variable to hold total circuit width
        self.sum_wire_heights = 0

        # em points of separation between circuit columns
        self.column_separation = 1

        # em points of separation between circuit wire
        self.wire_separation = 0

        # presence of "box" or "target" determines wire spacing
        self.has_box = False
        self.has_target = False
        self.layout = layout
        self.initial_state = initial_state
        self.reverse_bits = reverse_bits
        self.plot_barriers = plot_barriers

        #################################
        self.qubit_list = qubits
        self.ordered_bits = qubits + clbits
        self.cregs = {reg: reg.size for reg in cregs}

        self.bit_locations = {
            bit: {"register": register, "index": index}
            for register in cregs + qregs
            for index, bit in enumerate(register)
        }
        for index, bit in list(enumerate(qubits)) + list(enumerate(clbits)):
            if bit not in self.bit_locations:
                self.bit_locations[bit] = {"register": None, "index": index}

        self.cregbundle = cregbundle
        # If there is any custom instruction that uses clasiscal bits
        # then cregbundle is forced to be False.
        for layer in self.nodes:
            for node in layer:
                if node.op.name not in {"measure"} and node.cargs:
                    self.cregbundle = False

        self.cregs_bits = [self.bit_locations[bit]["register"] for bit in clbits]
        self.img_regs = {bit: ind for ind, bit in enumerate(self.ordered_bits)}

        if self.cregbundle:
            self.img_width = len(qubits) + len(self.cregs)
        else:
            self.img_width = len(self.img_regs)
        self.global_phase = global_phase

        self._style, _ = load_style(style)

    def latex(self):
        """Return LaTeX string representation of circuit."""

        self._initialize_latex_array()
        self._build_latex_array()
        header_1 = r"""\documentclass[border=2px]{standalone}
        """

        header_2 = r"""
\usepackage[braket, qm]{qcircuit}
\usepackage{graphicx}

\begin{document} 
"""
        header_scale = "\\scalebox{{{}}}".format(self.scale) + "{"

        qcircuit_line = r"""
\Qcircuit @C=%.1fem @R=%.1fem @!R { \\
"""
        output = io.StringIO()
        output.write(header_1)
        output.write(header_2)
        output.write(header_scale)
        if self.global_phase:
            output.write(
                r"""{$\mathrm{%s} \mathrm{%s}$}"""
                % ("global\\,phase:\\,", pi_check(self.global_phase, output="latex"))
            )
        output.write(qcircuit_line % (self.column_separation, self.wire_separation))
        for i in range(self.img_width):
            output.write("\t \t")
            for j in range(self.img_depth + 1):
                output.write(self._latex[i][j])
                if j != self.img_depth:
                    output.write(" & ")
                else:
                    output.write(r"\\ " + "\n")
        output.write(r"\\ " + "}}\n")
        output.write("\\end{document}")
        contents = output.getvalue()
        output.close()
        return contents

    def _initialize_latex_array(self):
        """Initialize qubit and clbit labels and set wire separation"""
        self.img_depth, self.sum_column_widths = self._get_image_depth()
        self.sum_wire_heights = self.img_width
        # choose the most compact wire spacing, while not squashing them
        if self.has_box:
            self.wire_separation = 0.2
        elif self.has_target:
            self.wire_separation = 0.8
        else:
            self.wire_separation = 1.0
        self._latex = [
            [
                "\\cw" if isinstance(self.ordered_bits[j], Clbit) else "\\qw"
                for _ in range(self.img_depth + 1)
            ]
            for j in range(self.img_width)
        ]
        self._latex.append([" "] * (self.img_depth + 1))
        if self.cregbundle:
            offset = 0
        for i in range(self.img_width):
            if isinstance(self.ordered_bits[i], Clbit):
                if self.cregbundle:
                    reg = self.bit_locations[self.ordered_bits[i + offset]]["register"]
                    label = reg.name + ":"
                    clbitsize = self.cregs[reg]
                    self._latex[i][1] = "\\lstick{/_{_{" + str(clbitsize) + "}}} \\cw"
                    offset += clbitsize - 1
                else:
                    label = self.bit_locations[self.ordered_bits[i]]["register"].name + "_{"
                    label += str(self.bit_locations[self.ordered_bits[i]]["index"]) + "}:"
                if self.initial_state:
                    label += "0"
                label += "}"
                self._latex[i][0] = "\\nghost{" + label + " & " + "\\lstick{" + label
            else:
                if self.layout is None:
                    label = " {{{}}}_{{{}}} : ".format(
                        self.bit_locations[self.ordered_bits[i]]["register"].name,
                        self.bit_locations[self.ordered_bits[i]]["index"],
                    )
                else:
                    bit_location = self.bit_locations[self.ordered_bits[i]]
                    if bit_location and self.layout[bit_location["index"]]:
                        virt_bit = self.layout[bit_location["index"]]
                        try:
                            virt_reg = next(
                                reg for reg in self.layout.get_registers() if virt_bit in reg
                            )
                            label = " {{{}}}_{{{}}}\\mapsto{{{}}} : ".format(
                                virt_reg.name, virt_reg[:].index(virt_bit), bit_location["index"]
                            )
                        except StopIteration:
                            label = "  {{{}}} : ".format(bit_location["index"])
                    else:
                        label = " {{{}}} : ".format(bit_location["index"])
                if self.initial_state:
                    label += "\\ket{{0}}"
                label += " }"
                self._latex[i][0] = "\\nghost{" + label + " & " + "\\lstick{" + label

    def _get_image_depth(self):
        """Get depth information for the circuit."""

        # wires in the beginning and end
        columns = 2
        if self.cregbundle and (
            self.nodes
            and self.nodes[0]
            and (self.nodes[0][0].op.name == "measure" or self.nodes[0][0].op.condition)
        ):
            columns += 1

        # Determine wire spacing before image depth
        max_column_widths = []
        for layer in self.nodes:
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
                    self.has_box = True
                elif op.name in target_gates:
                    self.has_target = True
                elif isinstance(op, ControlledGate):
                    self.has_box = True

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

        max_reg_name = 3
        for reg in self.ordered_bits:
            max_reg_name = max(max_reg_name, len(self.bit_locations[reg]["register"].name))
        sum_column_widths += 5 + max_reg_name / 3

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
        aspect_ratio = self.sum_wire_heights / self.sum_column_widths

        # choose a page margin so circuit is not cropped
        margin_factor = 1.5
        height = min(self.sum_wire_heights * margin_factor, beamer_limit)
        width = min(self.sum_column_widths * margin_factor, beamer_limit)

        # if too large, make it fit
        if height * width > pil_limit:
            height = min(np.sqrt(pil_limit * aspect_ratio), beamer_limit)
            width = min(np.sqrt(pil_limit / aspect_ratio), beamer_limit)

        # if too small, give it a minimum size
        height = max(height, 10)
        width = max(width, 10)

        return (height, width, self.scale)

    def _build_latex_array(self):
        """Returns an array of strings containing \\LaTeX for this circuit."""

        column = 1
        # Leave a column to display number of classical registers if needed
        if self.cregbundle and (
            self.nodes
            and self.nodes[0]
            and (self.nodes[0][0].op.name == "measure" or self.nodes[0][0].op.condition)
        ):
            column += 1

        for layer in self.nodes:
            num_cols_layer = 1

            for node in layer:
                op = node.op
                num_cols_op = 1
                if isinstance(op, Measure):
                    self._build_measure(node, column)

                elif op._directive:  # barrier, snapshot, etc.
                    self._build_barrier(node, column)

                else:
                    gate_text, _, _ = get_gate_ctrl_text(op, "latex", style=self._style)
                    gate_text += get_param_str(op, "latex", ndigits=4)
                    gate_text = generate_latex_label(gate_text)
                    wire_list = [self.img_regs[qarg] for qarg in node.qargs]
                    if node.cargs:
                        cwire_list = [self.img_regs[carg] for carg in node.cargs]
                    else:
                        cwire_list = []

                    if op.condition:
                        self._add_condition(op, wire_list, column)

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
        cwire_start = len(self.qubit_list)
        num_cols_op = 1
        if isinstance(op, (SwapGate, RZZGate)):
            num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
        else:
            wire_min = min(wire_list)
            wire_max = max(wire_list)
            if cwire_list and not self.cregbundle:
                wire_max = max(cwire_list)
            wire_ind = wire_list.index(wire_min)
            self._latex[wire_min][col] = (
                "\\multigate{%s}{%s}_" % (wire_max - wire_min, gate_text)
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
        if node.op.condition:
            raise exceptions.VisualizationError("If controlled measures currently not supported.")

        wire1 = self.img_regs[node.qargs[0]]
        if self.cregbundle:
            wire2 = len(self.qubit_list)
            cregindex = self.img_regs[node.cargs[0]] - wire2
            for creg_size in self.cregs.values():
                if cregindex >= creg_size:
                    cregindex -= creg_size
                    wire2 += 1
                else:
                    break
        else:
            wire2 = self.img_regs[node.cargs[0]]

        self._latex[wire1][col] = "\\meter"
        if self.cregbundle:
            self._latex[wire2][col] = "\\dstick{_{_{%s}}} \\cw \\cwx[-%s]" % (
                str(cregindex),
                str(wire2 - wire1),
            )
        else:
            self._latex[wire2][col] = "\\control \\cw \\cwx[-" + str(wire2 - wire1) + "]"

    def _build_barrier(self, node, col):
        """Build a partial or full barrier if plot_barriers set"""
        if self.plot_barriers:
            indexes = [self.img_regs[qarg] for qarg in node.qargs]
            indexes.sort()
            first = last = indexes[0]
            for index in indexes[1:]:
                if index - 1 == last:
                    last = index
                else:
                    pos = self.img_regs[self.qubit_list[first]]
                    self._latex[pos][col - 1] += " \\barrier[0em]{" + str(last - first) + "}"
                    self._latex[pos][col] = "\\qw"
                    first = last = index
            pos = self.img_regs[self.qubit_list[first]]
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
        # if_value - a bit string for the condition
        # cwire - the wire number for the first wire for the condition register
        #         or if cregbundle, wire number of the condition register itself
        # gap - the number of wires from cwire to the bottom gate qubit

        creg_size = self.cregs[op.condition[0]]
        if_value = format(op.condition[1], "b").zfill(creg_size)
        if not self.reverse_bits:
            if_value = if_value[::-1]

        cwire = len(self.qubit_list)
        iter_cregs = iter(list(self.cregs)) if self.cregbundle else iter(self.cregs_bits)
        for creg in iter_cregs:
            if creg == op.condition[0]:
                break
            cwire += 1

        gap = cwire - max(wire_list)
        if self.cregbundle:
            # Print the condition value at the bottom
            self._latex[cwire][col] = "\\dstick{_{_{=%s}}} \\cw \\cwx[-%s]" % (
                str(op.condition[1]),
                str(gap),
            )
        else:
            # Add the open and closed buttons to indicate the condition value
            for i in range(creg_size):
                control = "\\control" if if_value[i] == "1" else "\\controlo"
                self._latex[cwire + i][col] = f"{control} \\cw \\cwx[-" + str(gap) + "]"
                gap = 1

    def _truncate_float(self, matchobj, ndigits=4):
        """Truncate long floats."""
        if matchobj.group(0):
            return f"%.{ndigits}g" % float(matchobj.group(0))
        return ""
