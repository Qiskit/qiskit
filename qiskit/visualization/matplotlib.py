# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,inconsistent-return-statements

"""mpl circuit visualization backend."""

import re
from warnings import warn

import numpy as np


try:
    from pylatexenc.latex2text import LatexNodes2Text

    HAS_PYLATEX = True
except ImportError:
    HAS_PYLATEX = False

from qiskit.circuit import ControlledGate, Clbit
from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import (
    SwapGate,
    RZZGate,
    U1Gate,
    PhaseGate,
    XGate,
    ZGate,
)
from qiskit.extensions import Initialize
from qiskit.visualization.qcstyle import load_style
from qiskit.visualization.utils import (
    get_gate_ctrl_text,
    get_param_str,
    get_bit_label,
    matplotlib_close_if_inline,
)
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.exceptions import MissingOptionalLibraryError

# Default gate width and height
WID = 0.65
HIG = 0.65

PORDER_GATE = 5
PORDER_LINE = 3
PORDER_REGLINE = 2
PORDER_GRAY = 3
PORDER_TEXT = 6


class MatplotlibDrawer:
    """Matplotlib drawer class called from circuit_drawer"""

    _mathmode_regex = re.compile(r"(?<!\\)\$(.*)(?<!\\)\$")

    def __init__(
        self,
        qubits,
        clbits,
        nodes,
        scale=None,
        style=None,
        reverse_bits=False,
        plot_barriers=True,
        layout=None,
        fold=25,
        ax=None,
        initial_state=False,
        cregbundle=True,
        global_phase=None,
        qregs=None,
        cregs=None,
        calibrations=None,
    ):

        if not HAS_MATPLOTLIB:
            raise MissingOptionalLibraryError(
                libname="Matplotlib",
                name="MatplotlibDrawer",
                pip_install="pip install matplotlib",
            )
        from matplotlib import patches

        self._patches_mod = patches
        from matplotlib import pyplot as plt

        self._plt_mod = plt
        if not HAS_PYLATEX:
            raise MissingOptionalLibraryError(
                libname="pylatexenc",
                name="MatplotlibDrawer",
                pip_install="pip install pylatexenc",
            )

        # First load register and index info for the cregs and qregs,
        # then add any bits which don't have registers associated with them.
        self._bit_locations = {
            bit: {"register": register, "index": index}
            for register in cregs + qregs
            for index, bit in enumerate(register)
        }
        for index, bit in list(enumerate(qubits)) + list(enumerate(clbits)):
            if bit not in self._bit_locations:
                self._bit_locations[bit] = {"register": None, "index": index}

        self._qubit = qubits
        self._clbit = clbits
        self._qubit_dict = {}
        self._clbit_dict = {}
        self._nodes = nodes
        self._scale = 1.0 if scale is None else scale

        self._style, def_font_ratio = load_style(style)

        # If font/subfont ratio changes from default, have to scale width calculations for
        # subfont. Font change is auto scaled in the self._figure.set_size_inches call in draw()
        self._subfont_factor = self._style["sfs"] * def_font_ratio / self._style["fs"]

        self._reverse_bits = reverse_bits
        self._plot_barriers = plot_barriers
        self._layout = layout
        self._fold = fold
        if self._fold < 2:
            self._fold = -1

        if ax is None:
            self._user_ax = False
            self._figure = plt.figure()
            self._figure.patch.set_facecolor(color=self._style["bg"])
            self._ax = self._figure.add_subplot(111)
        else:
            self._user_ax = True
            self._ax = ax
            self._figure = ax.get_figure()
        self._ax.axis("off")
        self._ax.set_aspect("equal")
        self._ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        self._initial_state = initial_state
        self._cregbundle = cregbundle
        self._global_phase = global_phase
        self._calibrations = calibrations

        self._fs = self._style["fs"]
        self._sfs = self._style["sfs"]
        self._lwidth1 = 1.0
        self._lwidth15 = 1.5
        self._lwidth2 = 2.0
        self._x_offset = 0.0

        # _data per node with 'width', 'gate_text', 'raw_gate_text',
        # 'ctrl_text', 'param', q_xy', 'c_xy', and 'c_indxs'
        # and colors 'fc', 'ec', 'lc', 'sc', 'gt', and 'tc'
        self._data = {}
        self._layer_widths = []
        self._q_anchors = {}
        self._c_anchors = {}

        # _char_list for finding text_width of names, labels, and params
        self._char_list = {
            " ": (0.0958, 0.0583),
            "!": (0.1208, 0.0729),
            '"': (0.1396, 0.0875),
            "#": (0.2521, 0.1562),
            "$": (0.1917, 0.1167),
            "%": (0.2854, 0.1771),
            "&": (0.2333, 0.1458),
            "'": (0.0833, 0.0521),
            "(": (0.1167, 0.0729),
            ")": (0.1167, 0.0729),
            "*": (0.15, 0.0938),
            "+": (0.25, 0.1562),
            ",": (0.0958, 0.0583),
            "-": (0.1083, 0.0667),
            ".": (0.0958, 0.0604),
            "/": (0.1021, 0.0625),
            "0": (0.1875, 0.1167),
            "1": (0.1896, 0.1167),
            "2": (0.1917, 0.1188),
            "3": (0.1917, 0.1167),
            "4": (0.1917, 0.1188),
            "5": (0.1917, 0.1167),
            "6": (0.1896, 0.1167),
            "7": (0.1917, 0.1188),
            "8": (0.1896, 0.1188),
            "9": (0.1917, 0.1188),
            ":": (0.1021, 0.0604),
            ";": (0.1021, 0.0604),
            "<": (0.25, 0.1542),
            "=": (0.25, 0.1562),
            ">": (0.25, 0.1542),
            "?": (0.1583, 0.0979),
            "@": (0.2979, 0.1854),
            "A": (0.2062, 0.1271),
            "B": (0.2042, 0.1271),
            "C": (0.2083, 0.1292),
            "D": (0.2312, 0.1417),
            "E": (0.1875, 0.1167),
            "F": (0.1708, 0.1062),
            "G": (0.2312, 0.1438),
            "H": (0.225, 0.1396),
            "I": (0.0875, 0.0542),
            "J": (0.0875, 0.0542),
            "K": (0.1958, 0.1208),
            "L": (0.1667, 0.1042),
            "M": (0.2583, 0.1604),
            "N": (0.225, 0.1396),
            "O": (0.2354, 0.1458),
            "P": (0.1812, 0.1125),
            "Q": (0.2354, 0.1458),
            "R": (0.2083, 0.1292),
            "S": (0.1896, 0.1188),
            "T": (0.1854, 0.1125),
            "U": (0.2208, 0.1354),
            "V": (0.2062, 0.1271),
            "W": (0.2958, 0.1833),
            "X": (0.2062, 0.1271),
            "Y": (0.1833, 0.1125),
            "Z": (0.2042, 0.1271),
            "[": (0.1167, 0.075),
            "\\": (0.1021, 0.0625),
            "]": (0.1167, 0.0729),
            "^": (0.2521, 0.1562),
            "_": (0.1521, 0.0938),
            "`": (0.15, 0.0938),
            "a": (0.1854, 0.1146),
            "b": (0.1917, 0.1167),
            "c": (0.1646, 0.1021),
            "d": (0.1896, 0.1188),
            "e": (0.1854, 0.1146),
            "f": (0.1042, 0.0667),
            "g": (0.1896, 0.1188),
            "h": (0.1896, 0.1188),
            "i": (0.0854, 0.0521),
            "j": (0.0854, 0.0521),
            "k": (0.1729, 0.1083),
            "l": (0.0854, 0.0521),
            "m": (0.2917, 0.1812),
            "n": (0.1896, 0.1188),
            "o": (0.1833, 0.1125),
            "p": (0.1917, 0.1167),
            "q": (0.1896, 0.1188),
            "r": (0.125, 0.0771),
            "s": (0.1562, 0.0958),
            "t": (0.1167, 0.0729),
            "u": (0.1896, 0.1188),
            "v": (0.1771, 0.1104),
            "w": (0.2458, 0.1521),
            "x": (0.1771, 0.1104),
            "y": (0.1771, 0.1104),
            "z": (0.1562, 0.0979),
            "{": (0.1917, 0.1188),
            "|": (0.1, 0.0604),
            "}": (0.1896, 0.1188),
        }

    def draw(self, filename=None, verbose=False):
        """Main entry point to 'matplotlib' ('mpl') drawer. Called from
        ``visualization.circuit_drawer`` and from ``QuantumCircuit.draw`` through circuit_drawer.
        """
        # All information for the drawing is first loaded into self._data for the gates and into
        # self._qubit_dict and self._clbit_dict for the qubits, clbits, and wires,
        # followed by the coordinates for each gate.

        # get layer widths
        self._get_layer_widths()

        # load the _qubit_dict and _clbit_dict with register info
        n_lines = self._get_bit_labels()

        # load the coordinates for each gate and compute number of folds
        max_anc = self._get_coords(n_lines)
        num_folds = max(0, max_anc - 1) // self._fold if self._fold > 0 else 0

        # The window size limits are computed, followed by one of the four possible ways
        # of scaling the drawing.

        # compute the window size
        if max_anc > self._fold > 0:
            xmax = self._fold + self._x_offset + 0.1
            ymax = (num_folds + 1) * (n_lines + 1) - 1
        else:
            x_incr = 0.4 if not self._nodes else 0.9
            xmax = max_anc + 1 + self._x_offset - x_incr
            ymax = n_lines

        xl = -self._style["margin"][0]
        xr = xmax + self._style["margin"][1]
        yb = -ymax - self._style["margin"][2] + 0.5
        yt = self._style["margin"][3] + 0.5
        self._ax.set_xlim(xl, xr)
        self._ax.set_ylim(yb, yt)

        # update figure size and, for backward compatibility,
        # need to scale by a default value equal to (self._fs * 3.01 / 72 / 0.65)
        base_fig_w = (xr - xl) * 0.8361111
        base_fig_h = (yt - yb) * 0.8361111
        scale = self._scale

        # if user passes in an ax, this size takes priority over any other settings
        if self._user_ax:
            # from stackoverflow #19306510, get the bbox size for the ax and then reset scale
            bbox = self._ax.get_window_extent().transformed(self._figure.dpi_scale_trans.inverted())
            scale = bbox.width / base_fig_w / 0.8361111

        # if scale not 1.0, use this scale factor
        elif self._scale != 1.0:
            self._figure.set_size_inches(base_fig_w * self._scale, base_fig_h * self._scale)

        # if "figwidth" style param set, use this to scale
        elif self._style["figwidth"] > 0.0:
            # in order to get actual inches, need to scale by factor
            adj_fig_w = self._style["figwidth"] * 1.282736
            self._figure.set_size_inches(adj_fig_w, adj_fig_w * base_fig_h / base_fig_w)
            scale = adj_fig_w / base_fig_w

        # otherwise, display default size
        else:
            self._figure.set_size_inches(base_fig_w, base_fig_h)

        # drawing will scale with 'set_size_inches', but fonts and linewidths do not
        if scale != 1.0:
            self._fs *= scale
            self._sfs *= scale
            self._lwidth1 = 1.0 * scale
            self._lwidth15 = 1.5 * scale
            self._lwidth2 = 2.0 * scale

        # Once the scaling factor has been determined, the global phase, register names
        # and numbers, wires, and gates are drawn
        if self._global_phase:
            self._plt_mod.text(
                xl, yt, "Global Phase: %s" % pi_check(self._global_phase, output="mpl")
            )
        self._draw_regs_wires(num_folds, xmax, n_lines, max_anc)
        self._draw_ops(verbose)

        if filename:
            self._figure.savefig(
                filename,
                dpi=self._style["dpi"],
                bbox_inches="tight",
                facecolor=self._figure.get_facecolor(),
            )
        if not self._user_ax:
            matplotlib_close_if_inline(self._figure)
            return self._figure

    def _get_layer_widths(self):
        """Compute the layer_widths for the layers"""
        for layer in self._nodes:
            widest_box = WID
            for node in layer:
                op = node.op
                if self._cregbundle and node.cargs and not isinstance(op, Measure):
                    self._cregbundle = False
                    warn(
                        "Cregbundle set to False since an instruction needs to refer"
                        " to individual classical wire",
                        RuntimeWarning,
                        2,
                    )
                self._data[node] = {}
                self._data[node]["width"] = WID
                num_ctrl_qubits = 0 if not hasattr(op, "num_ctrl_qubits") else op.num_ctrl_qubits
                if op._directive or isinstance(op, Measure):
                    self._data[node]["raw_gate_text"] = op.name
                    continue

                base_type = None if not hasattr(op, "base_gate") else op.base_gate
                gate_text, ctrl_text, raw_gate_text = get_gate_ctrl_text(
                    op, "mpl", style=self._style, calibrations=self._calibrations
                )
                self._data[node]["gate_text"] = gate_text
                self._data[node]["ctrl_text"] = ctrl_text
                self._data[node]["raw_gate_text"] = raw_gate_text
                self._data[node]["param"] = ""

                # if single qubit, no params, and no labels, layer_width is 1
                if (
                    (len(node.qargs) - num_ctrl_qubits) == 1
                    and len(gate_text) < 3
                    and (not hasattr(op, "params") or len(op.params) == 0)
                    and ctrl_text is None
                ):
                    continue

                if isinstance(op, SwapGate) or isinstance(base_type, SwapGate):
                    continue

                # small increments at end of the 3 _get_text_width calls are for small
                # spacing adjustments between gates
                ctrl_width = self._get_text_width(ctrl_text, fontsize=self._sfs) - 0.05

                # get param_width, but 0 for gates with array params
                if (
                    hasattr(op, "params")
                    and len(op.params) > 0
                    and not any(isinstance(param, np.ndarray) for param in op.params)
                ):
                    param = get_param_str(op, "mpl", ndigits=3)
                    if isinstance(op, Initialize):
                        param = f"$[{param.replace('$', '')}]$"
                    self._data[node]["param"] = param
                    raw_param_width = self._get_text_width(param, fontsize=self._sfs, param=True)
                    param_width = raw_param_width + 0.08
                else:
                    param_width = raw_param_width = 0.0

                # get gate_width for sidetext symmetric gates
                if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
                    if isinstance(base_type, PhaseGate):
                        gate_text = "P"
                    raw_gate_width = (
                        self._get_text_width(gate_text + " ()", fontsize=self._sfs)
                        + raw_param_width
                    )
                    gate_width = (raw_gate_width + 0.08) * 1.58

                # otherwise, standard gate or multiqubit gate
                else:
                    raw_gate_width = self._get_text_width(gate_text, fontsize=self._fs)
                    gate_width = raw_gate_width + 0.10
                    # add .21 for the qubit numbers on the left of the multibit gates
                    if len(node.qargs) - num_ctrl_qubits > 1:
                        gate_width += 0.21

                box_width = max(gate_width, ctrl_width, param_width, WID)
                if box_width > widest_box:
                    widest_box = box_width
                self._data[node]["width"] = max(raw_gate_width, raw_param_width)

            self._layer_widths.append(int(widest_box) + 1)

    def _get_bit_labels(self):
        """Get all the info for drawing reg names and numbers"""
        longest_bit_label_width = 0
        n_lines = 0
        initial_qbit = " |0>" if self._initial_state else ""
        initial_cbit = " 0" if self._initial_state else ""

        def _fix_double_script(bit_label):
            words = bit_label.split(" ")
            words = [word.replace("_", r"\_") if word.count("_") > 1 else word for word in words]
            words = [
                word.replace("^", r"\^{\ }") if word.count("^") > 1 else word for word in words
            ]
            bit_label = " ".join(words).replace(" ", "\\;")
            return bit_label

        # quantum register
        for ii, reg in enumerate(self._qubit):
            register = self._bit_locations[reg]["register"]
            index = self._bit_locations[reg]["index"]
            qubit_label = get_bit_label("mpl", register, index, qubit=True, layout=self._layout)
            qubit_label = "$" + _fix_double_script(qubit_label) + "$" + initial_qbit

            text_width = self._get_text_width(qubit_label, self._fs) * 1.15
            if text_width > longest_bit_label_width:
                longest_bit_label_width = text_width
            pos = -ii
            self._qubit_dict[ii] = {
                "y": pos,
                "bit_label": qubit_label,
                "index": index,
                "register": register,
            }
            n_lines += 1

        # classical register
        if self._clbit:
            prev_creg = None
            idx = 0
            pos = y_off = -len(self._qubit) + 1
            for ii, reg in enumerate(self._clbit):
                register = self._bit_locations[reg]["register"]
                index = self._bit_locations[reg]["index"]
                if register is None or not self._cregbundle or prev_creg != register:
                    n_lines += 1
                    idx += 1

                prev_creg = register
                clbit_label = get_bit_label(
                    "mpl", register, index, qubit=False, cregbundle=self._cregbundle
                )
                clbit_label = _fix_double_script(clbit_label)
                if register is None or not self._cregbundle:
                    clbit_label = "$" + clbit_label + "$"
                clbit_label += initial_cbit

                text_width = self._get_text_width(clbit_label, self._fs) * 1.15
                if text_width > longest_bit_label_width:
                    longest_bit_label_width = text_width
                pos = y_off - idx
                self._clbit_dict[ii] = {
                    "y": pos,
                    "bit_label": clbit_label,
                    "index": index,
                    "register": register,
                }
        self._x_offset = -1.2 + longest_bit_label_width
        return n_lines

    def _get_coords(self, n_lines):
        """Load all the coordinate info needed to place the gates on the drawing"""

        # create the anchor arrays
        for key, qubit in self._qubit_dict.items():
            self._q_anchors[key] = Anchor(reg_num=n_lines, yind=qubit["y"], fold=self._fold)
        for key, clbit in self._clbit_dict.items():
            self._c_anchors[key] = Anchor(reg_num=n_lines, yind=clbit["y"], fold=self._fold)

        # get all the necessary coordinates for placing gates on the wires
        prev_anc = -1
        for i, layer in enumerate(self._nodes):
            layer_width = self._layer_widths[i]
            this_anc = prev_anc + 1
            for node in layer:
                # get qubit index
                q_indxs = []
                for qarg in node.qargs:
                    for index, reg in self._qubit_dict.items():
                        if (
                            reg["register"] == self._bit_locations[qarg]["register"]
                            and reg["index"] == self._bit_locations[qarg]["index"]
                        ):
                            q_indxs.append(index)
                            break

                # get clbit index
                c_indxs = []
                for carg in node.cargs:
                    for index, reg in self._clbit_dict.items():
                        if (
                            reg["register"] == self._bit_locations[carg]["register"]
                            and reg["index"] == self._bit_locations[carg]["index"]
                        ):
                            c_indxs.append(index)
                            break

                # only add the gate to the anchors if it is going to be plotted.
                if self._plot_barriers or not node.op._directive:
                    for ii in q_indxs:
                        self._q_anchors[ii].set_index(this_anc, layer_width)

                # qubit coordinate
                self._data[node]["q_xy"] = [
                    self._q_anchors[ii].plot_coord(this_anc, layer_width, self._x_offset)
                    for ii in q_indxs
                ]
                # clbit coordinate
                self._data[node]["c_xy"] = [
                    self._c_anchors[ii].plot_coord(this_anc, layer_width, self._x_offset)
                    for ii in c_indxs
                ]
                # update index based on the value from plotting
                this_anc = self._q_anchors[q_indxs[0]].gate_anchor
                self._data[node]["c_indxs"] = c_indxs

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self._plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = -1 if all(nd.op._directive for nd in layer) else 0
            prev_anc = this_anc + layer_width + barrier_offset - 1

        anchors = [self._q_anchors[ii].get_index() for ii in self._qubit_dict]
        return max(anchors) if anchors else 0

    def _get_text_width(self, text, fontsize, param=False):
        """Compute the width of a string in the default font"""
        if not text:
            return 0.0

        math_mode_match = self._mathmode_regex.search(text)
        num_underscores = 0
        num_carets = 0
        if math_mode_match:
            math_mode_text = math_mode_match.group(1)
            num_underscores = math_mode_text.count("_")
            num_carets = math_mode_text.count("^")
        text = LatexNodes2Text().latex_to_text(text.replace("$$", ""))

        # If there are subscripts or superscripts in mathtext string
        # we need to account for that spacing by manually removing
        # from text string for text length
        if num_underscores:
            text = text.replace("_", "", num_underscores)
        if num_carets:
            text = text.replace("^", "", num_carets)

        # This changes hyphen to + to match width of math mode minus sign.
        if param:
            text = text.replace("-", "+")

        f = 0 if fontsize == self._fs else 1
        sum_text = 0.0
        for c in text:
            try:
                sum_text += self._char_list[c][f]
            except KeyError:
                # if non-ASCII char, use width of 'c', an average size
                sum_text += self._char_list["c"][f]
        if f == 1:
            sum_text *= self._subfont_factor
        return sum_text

    def _draw_regs_wires(self, num_folds, xmax, n_lines, max_anc):
        """Draw the register names and numbers, wires, and vertical lines at the ends"""

        for fold_num in range(num_folds + 1):
            # quantum registers
            for qubit in self._qubit_dict.values():
                qubit_label = qubit["bit_label"]
                y = qubit["y"] - fold_num * (n_lines + 1)
                self._ax.text(
                    self._x_offset - 0.2,
                    y,
                    qubit_label,
                    ha="right",
                    va="center",
                    fontsize=1.25 * self._fs,
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
                # draw the qubit wire
                self._line([self._x_offset, y], [xmax, y], zorder=PORDER_REGLINE)

            # classical registers
            this_clbit_dict = {}
            for clbit in self._clbit_dict.values():
                clbit_label = clbit["bit_label"]
                clbit_reg = clbit["register"]
                y = clbit["y"] - fold_num * (n_lines + 1)
                if y not in this_clbit_dict.keys():
                    this_clbit_dict[y] = {"val": 1, "bit_label": clbit_label, "register": clbit_reg}
                else:
                    this_clbit_dict[y]["val"] += 1

            for y, this_clbit in this_clbit_dict.items():
                # cregbundle
                if self._cregbundle and this_clbit["register"] is not None:
                    self._ax.plot(
                        [self._x_offset + 0.2, self._x_offset + 0.3],
                        [y - 0.1, y + 0.1],
                        color=self._style["cc"],
                        zorder=PORDER_LINE,
                    )
                    self._ax.text(
                        self._x_offset + 0.1,
                        y + 0.1,
                        str(this_clbit["val"]),
                        ha="left",
                        va="bottom",
                        fontsize=0.8 * self._fs,
                        color=self._style["tc"],
                        clip_on=True,
                        zorder=PORDER_TEXT,
                    )
                self._ax.text(
                    self._x_offset - 0.2,
                    y,
                    this_clbit["bit_label"],
                    ha="right",
                    va="center",
                    fontsize=1.25 * self._fs,
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
                # draw the clbit wire
                self._line(
                    [self._x_offset, y],
                    [xmax, y],
                    lc=self._style["cc"],
                    ls=self._style["cline"],
                    zorder=PORDER_REGLINE,
                )

            # lf vertical line at either end
            feedline_r = num_folds > 0 and num_folds > fold_num
            feedline_l = fold_num > 0
            if feedline_l or feedline_r:
                xpos_l = self._x_offset - 0.01
                xpos_r = self._fold + self._x_offset + 0.1
                ypos1 = -fold_num * (n_lines + 1)
                ypos2 = -(fold_num + 1) * (n_lines) - fold_num + 1
                if feedline_l:
                    self._ax.plot(
                        [xpos_l, xpos_l],
                        [ypos1, ypos2],
                        color=self._style["lc"],
                        linewidth=self._lwidth15,
                        zorder=PORDER_LINE,
                    )
                if feedline_r:
                    self._ax.plot(
                        [xpos_r, xpos_r],
                        [ypos1, ypos2],
                        color=self._style["lc"],
                        linewidth=self._lwidth15,
                        zorder=PORDER_LINE,
                    )

        # draw anchor index number
        if self._style["index"]:
            for layer_num in range(max_anc):
                if self._fold > 0:
                    x_coord = layer_num % self._fold + self._x_offset + 0.53
                    y_coord = -(layer_num // self._fold) * (n_lines + 1) + 0.65
                else:
                    x_coord = layer_num + self._x_offset + 0.53
                    y_coord = 0.65
                self._ax.text(
                    x_coord,
                    y_coord,
                    str(layer_num + 1),
                    ha="center",
                    va="center",
                    fontsize=self._sfs,
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )

    def _draw_ops(self, verbose=False):
        """Draw the gates in the circuit"""
        prev_anc = -1
        for i, layer in enumerate(self._nodes):
            layer_width = self._layer_widths[i]
            this_anc = prev_anc + 1

            # draw the gates in this layer
            for node in layer:
                op = node.op
                self._get_colors(node)

                if verbose:
                    print(op)

                # add conditional
                if op.condition:
                    cond_xy = [
                        self._c_anchors[ii].plot_coord(this_anc, layer_width, self._x_offset)
                        for ii in self._clbit_dict
                    ]
                    self._condition(node, cond_xy)

                # draw measure
                if isinstance(op, Measure):
                    self._measure(node)

                # draw barriers, snapshots, etc.
                elif op._directive:
                    if self._plot_barriers:
                        self._barrier(node)

                # draw single qubit gates
                elif len(self._data[node]["q_xy"]) == 1 and not node.cargs:
                    self._gate(node)

                # draw controlled gates
                elif isinstance(op, ControlledGate):
                    self._control_gate(node)

                # draw multi-qubit gate as final default
                else:
                    self._multiqubit_gate(node)

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self._plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = -1 if all(nd.op._directive for nd in layer) else 0

            prev_anc = this_anc + layer_width + barrier_offset - 1

    def _get_colors(self, node):
        """Get all the colors needed for drawing the circuit"""
        op = node.op
        base_name = None if not hasattr(op, "base_gate") else op.base_gate.name
        color = None
        if self._data[node]["raw_gate_text"] in self._style["dispcol"]:
            color = self._style["dispcol"][self._data[node]["raw_gate_text"]]
        elif op.name in self._style["dispcol"]:
            color = self._style["dispcol"][op.name]
        if color is not None:
            # Backward compatibility for style dict using 'displaycolor' with
            # gate color and no text color, so test for str first
            if isinstance(color, str):
                fc = color
                gt = self._style["gt"]
            else:
                fc = color[0]
                gt = color[1]
        # Treat special case of classical gates in iqx style by making all
        # controlled gates of x, dcx, and swap the classical gate color
        elif self._style["name"] == "iqx" and base_name in ["x", "dcx", "swap"]:
            color = self._style["dispcol"][base_name]
            if isinstance(color, str):
                fc = color
                gt = self._style["gt"]
            else:
                fc = color[0]
                gt = color[1]
        else:
            fc = self._style["gc"]
            gt = self._style["gt"]

        if self._style["name"] == "bw":
            ec = self._style["ec"]
            lc = self._style["lc"]
        else:
            ec = fc
            lc = fc
        # Subtext needs to be same color as gate text
        sc = gt
        self._data[node]["fc"] = fc
        self._data[node]["ec"] = ec
        self._data[node]["gt"] = gt
        self._data[node]["tc"] = self._style["tc"]
        self._data[node]["sc"] = sc
        self._data[node]["lc"] = lc

    def _condition(self, node, cond_xy):
        """Add a conditional to a gate"""
        cond_is_bit = bool(isinstance(node.op.condition[0], Clbit))
        mask = 0
        qubit_b = min(self._data[node]["q_xy"], key=lambda xy: xy[1])
        if cond_is_bit:
            for index, cbit in enumerate(self._clbit):
                if cbit == node.op.condition[0]:
                    mask = 1 << index
                    break
        else:
            for index, cbit in enumerate(self._clbit):
                if self._bit_locations[cbit]["register"] == node.op.condition[0]:
                    mask |= 1 << index
        val = node.op.condition[1]

        # cbit list to consider
        fmt_c = f"{{:0{len(cond_xy)}b}}"
        cmask = list(fmt_c.format(mask))[::-1]

        # value
        fmt_v = f"{{:0{cmask.count('1')}b}}"
        vlist = list(fmt_v.format(val))
        if not self._reverse_bits:
            vlist = vlist[::-1]

        # plot conditionals
        v_ind = 0
        xy_plot = []
        for xy, m in zip(cond_xy, cmask):
            if m == "1":
                if xy not in xy_plot:
                    if vlist[v_ind] == "1" or self._cregbundle:
                        fc = self._style["lc"]
                    else:
                        fc = self._style["bg"]
                    box = self._patches_mod.Circle(
                        xy=xy,
                        radius=WID * 0.15,
                        fc=fc,
                        ec=self._style["lc"],
                        linewidth=self._lwidth15,
                        zorder=PORDER_GATE,
                    )
                    self._ax.add_patch(box)
                    xy_plot.append(xy)
                v_ind += 1
        clbit_b = min(xy_plot, key=lambda xy: xy[1])
        xpos, ypos = clbit_b
        if cond_is_bit and self._cregbundle:
            cond_reg = self._bit_locations[node.op.condition[0]]["register"]
            ctrl_bit = self._bit_locations[node.op.condition[0]]["index"]
            label = "%s_%s=%s" % (cond_reg.name, ctrl_bit, hex(val))
        else:
            label = hex(val)
        if isinstance(node.op, Measure):
            xpos += 0.3
        self._ax.text(
            xpos,
            ypos - 0.3 * HIG,
            label,
            ha="center",
            va="top",
            fontsize=self._sfs,
            color=self._style["tc"],
            clip_on=True,
            zorder=PORDER_TEXT,
        )
        self._line(qubit_b, clbit_b, lc=self._style["cc"], ls=self._style["cline"])

    def _measure(self, node):
        """Draw the measure symbol and the line to the clbit"""
        qx, qy = self._data[node]["q_xy"][0]
        cx, cy = self._data[node]["c_xy"][0]
        cid = self._clbit_dict[self._data[node]["c_indxs"][0]]["index"]

        # draw gate box
        self._gate(node)

        # add measure symbol
        arc = self._patches_mod.Arc(
            xy=(qx, qy - 0.15 * HIG),
            width=WID * 0.7,
            height=HIG * 0.7,
            theta1=0,
            theta2=180,
            fill=False,
            ec=self._data[node]["gt"],
            linewidth=self._lwidth2,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(arc)
        self._ax.plot(
            [qx, qx + 0.35 * WID],
            [qy - 0.15 * HIG, qy + 0.20 * HIG],
            color=self._data[node]["gt"],
            linewidth=self._lwidth2,
            zorder=PORDER_GATE,
        )
        # arrow
        self._line(
            self._data[node]["q_xy"][0],
            [cx, cy + 0.35 * WID],
            lc=self._style["cc"],
            ls=self._style["cline"],
        )
        arrowhead = self._patches_mod.Polygon(
            (
                (cx - 0.20 * WID, cy + 0.35 * WID),
                (cx + 0.20 * WID, cy + 0.35 * WID),
                (cx, cy + 0.04),
            ),
            fc=self._style["cc"],
            ec=None,
        )
        self._ax.add_artist(arrowhead)
        # target
        if self._cregbundle:
            self._ax.text(
                cx + 0.25,
                cy + 0.1,
                str(cid),
                ha="left",
                va="bottom",
                fontsize=0.8 * self._fs,
                color=self._style["tc"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _barrier(self, node):
        """Draw a barrier"""
        for xy in self._data[node]["q_xy"]:
            xpos, ypos = xy
            self._ax.plot(
                [xpos, xpos],
                [ypos + 0.5, ypos - 0.5],
                linewidth=self._lwidth1,
                linestyle="dashed",
                color=self._style["lc"],
                zorder=PORDER_TEXT,
            )
            box = self._patches_mod.Rectangle(
                xy=(xpos - (0.3 * WID), ypos - 0.5),
                width=0.6 * WID,
                height=1,
                fc=self._style["bc"],
                ec=None,
                alpha=0.6,
                linewidth=self._lwidth15,
                zorder=PORDER_GRAY,
            )
            self._ax.add_patch(box)

    def _gate(self, node, xy=None):
        """Draw a 1-qubit gate"""
        if xy is None:
            xy = self._data[node]["q_xy"][0]
        xpos, ypos = xy
        wid = max(self._data[node]["width"], WID)

        box = self._patches_mod.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG),
            width=wid,
            height=HIG,
            fc=self._data[node]["fc"],
            ec=self._data[node]["ec"],
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        if "gate_text" in self._data[node]:
            gate_ypos = ypos
            if "param" in self._data[node] and self._data[node]["param"] != "":
                gate_ypos = ypos + 0.15 * HIG
                self._ax.text(
                    xpos,
                    ypos - 0.3 * HIG,
                    self._data[node]["param"],
                    ha="center",
                    va="center",
                    fontsize=self._sfs,
                    color=self._data[node]["sc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
            self._ax.text(
                xpos,
                gate_ypos,
                self._data[node]["gate_text"],
                ha="center",
                va="center",
                fontsize=self._fs,
                color=self._data[node]["gt"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _multiqubit_gate(self, node, xy=None):
        """Draw a gate covering more than one qubit"""
        op = node.op
        if xy is None:
            xy = self._data[node]["q_xy"]

        # Swap gate
        if isinstance(op, SwapGate):
            self._swap(xy, node, self._data[node]["lc"])
            return

        # RZZ Gate
        elif isinstance(op, RZZGate):
            self._symmetric_gate(node, RZZGate)
            return

        c_xy = self._data[node]["c_xy"]
        xpos = min(x[0] for x in xy)
        ypos = min(y[1] for y in xy)
        ypos_max = max(y[1] for y in xy)
        if c_xy:
            cxpos = min(x[0] for x in c_xy)
            cypos = min(y[1] for y in c_xy)
            ypos = min(ypos, cypos)

        wid = max(self._data[node]["width"] + 0.21, WID)

        qubit_span = abs(ypos) - abs(ypos_max) + 1
        height = HIG + (qubit_span - 1)
        box = self._patches_mod.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG),
            width=wid,
            height=height,
            fc=self._data[node]["fc"],
            ec=self._data[node]["ec"],
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        # annotate inputs
        for bit, y in enumerate([x[1] for x in xy]):
            self._ax.text(
                xpos + 0.07 - 0.5 * wid,
                y,
                str(bit),
                ha="left",
                va="center",
                fontsize=self._fs,
                color=self._data[node]["gt"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )
        if c_xy:
            # annotate classical inputs
            for bit, y in enumerate([x[1] for x in c_xy]):
                self._ax.text(
                    cxpos + 0.07 - 0.5 * wid,
                    y,
                    str(bit),
                    ha="left",
                    va="center",
                    fontsize=self._fs,
                    color=self._data[node]["gt"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
        if "gate_text" in self._data[node] and self._data[node]["gate_text"] != "":
            gate_ypos = ypos + 0.5 * (qubit_span - 1)
            if "param" in self._data[node] and self._data[node]["param"] != "":
                gate_ypos = ypos + 0.4 * height
                self._ax.text(
                    xpos + 0.11,
                    ypos + 0.2 * height,
                    self._data[node]["param"],
                    ha="center",
                    va="center",
                    fontsize=self._sfs,
                    color=self._data[node]["sc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
            self._ax.text(
                xpos + 0.11,
                gate_ypos,
                self._data[node]["gate_text"],
                ha="center",
                va="center",
                fontsize=self._fs,
                color=self._data[node]["gt"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _control_gate(self, node):
        """Draw a controlled gate"""
        op = node.op
        base_type = None if not hasattr(op, "base_gate") else op.base_gate
        xy = self._data[node]["q_xy"]
        qubit_b = min(xy, key=lambda xy: xy[1])
        qubit_t = max(xy, key=lambda xy: xy[1])
        num_ctrl_qubits = op.num_ctrl_qubits
        num_qargs = len(xy) - num_ctrl_qubits
        self._set_ctrl_bits(
            op.ctrl_state,
            num_ctrl_qubits,
            xy,
            ec=self._data[node]["ec"],
            tc=self._data[node]["tc"],
            text=self._data[node]["ctrl_text"],
            qargs=node.qargs,
        )
        self._line(qubit_b, qubit_t, lc=self._data[node]["lc"])

        if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, ZGate, RZZGate)):
            self._symmetric_gate(node, base_type)

        elif num_qargs == 1 and isinstance(base_type, XGate):
            tgt_color = self._style["dispcol"]["target"]
            tgt = tgt_color if isinstance(tgt_color, str) else tgt_color[0]
            self._x_tgt_qubit(xy[num_ctrl_qubits], ec=self._data[node]["ec"], ac=tgt)

        elif num_qargs == 1:
            self._gate(node, xy[num_ctrl_qubits:][0])

        elif isinstance(base_type, SwapGate):
            self._swap(xy[num_ctrl_qubits:], node, self._data[node]["lc"])

        else:
            self._multiqubit_gate(node, xy[num_ctrl_qubits:])

    def _set_ctrl_bits(
        self, ctrl_state, num_ctrl_qubits, qbit, ec=None, tc=None, text="", qargs=None
    ):
        """Determine which qubits are controls and whether they are open or closed"""
        # place the control label at the top or bottom of controls
        if text:
            qlist = [self._bit_locations[qubit]["index"] for qubit in qargs]
            ctbits = qlist[:num_ctrl_qubits]
            qubits = qlist[num_ctrl_qubits:]
            max_ctbit = max(ctbits)
            min_ctbit = min(ctbits)
            top = min(qubits) > min_ctbit

        # display the control qubits as open or closed based on ctrl_state
        cstate = f"{ctrl_state:b}".rjust(num_ctrl_qubits, "0")[::-1]
        for i in range(num_ctrl_qubits):
            fc_open_close = ec if cstate[i] == "1" else self._style["bg"]
            text_top = None
            if text:
                if top and qlist[i] == min_ctbit:
                    text_top = True
                elif not top and qlist[i] == max_ctbit:
                    text_top = False
            self._ctrl_qubit(qbit[i], fc=fc_open_close, ec=ec, tc=tc, text=text, text_top=text_top)

    def _ctrl_qubit(self, xy, fc=None, ec=None, tc=None, text="", text_top=None):
        """Draw a control circle and if top or bottom control, draw control label"""
        xpos, ypos = xy
        box = self._patches_mod.Circle(
            xy=(xpos, ypos),
            radius=WID * 0.15,
            fc=fc,
            ec=ec,
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        # adjust label height according to number of lines of text
        label_padding = 0.7
        if text is not None:
            text_lines = text.count("\n")
            if not text.endswith("(cal)\n"):
                for _ in range(text_lines):
                    label_padding += 0.3

        if text_top is None:
            return

        # display the control label at the top or bottom if there is one
        ctrl_ypos = ypos + label_padding * HIG if text_top else ypos - 0.3 * HIG
        self._ax.text(
            xpos,
            ctrl_ypos,
            text,
            ha="center",
            va="top",
            fontsize=self._sfs,
            color=tc,
            clip_on=True,
            zorder=PORDER_TEXT,
        )

    def _x_tgt_qubit(self, xy, ec=None, ac=None):
        """Draw the cnot target symbol"""
        linewidth = self._lwidth2
        xpos, ypos = xy
        box = self._patches_mod.Circle(
            xy=(xpos, ypos),
            radius=HIG * 0.35,
            fc=ec,
            ec=ec,
            linewidth=linewidth,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        # add '+' symbol
        self._ax.plot(
            [xpos, xpos],
            [ypos - 0.2 * HIG, ypos + 0.2 * HIG],
            color=ac,
            linewidth=linewidth,
            zorder=PORDER_GATE + 1,
        )
        self._ax.plot(
            [xpos - 0.2 * HIG, xpos + 0.2 * HIG],
            [ypos, ypos],
            color=ac,
            linewidth=linewidth,
            zorder=PORDER_GATE + 1,
        )

    def _symmetric_gate(self, node, base_type):
        """Draw symmetric gates for cz, cu1, cp, and rzz"""
        op = node.op
        xy = self._data[node]["q_xy"]
        qubit_b = min(xy, key=lambda xy: xy[1])
        qubit_t = max(xy, key=lambda xy: xy[1])
        base_type = None if not hasattr(op, "base_gate") else op.base_gate
        ec = self._data[node]["ec"]
        tc = self._data[node]["tc"]
        lc = self._data[node]["lc"]

        # cz and mcz gates
        if not isinstance(op, ZGate) and isinstance(base_type, ZGate):
            num_ctrl_qubits = op.num_ctrl_qubits
            self._ctrl_qubit(xy[-1], fc=ec, ec=ec, tc=tc)
            self._line(qubit_b, qubit_t, lc=lc, zorder=PORDER_LINE + 1)

        # cu1, cp, rzz, and controlled rzz gates (sidetext gates)
        elif isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
            num_ctrl_qubits = 0 if isinstance(op, RZZGate) else op.num_ctrl_qubits
            gate_text = "P" if isinstance(base_type, PhaseGate) else self._data[node]["gate_text"]

            self._ctrl_qubit(xy[num_ctrl_qubits], fc=ec, ec=ec, tc=tc)
            if not isinstance(base_type, (U1Gate, PhaseGate)):
                self._ctrl_qubit(xy[num_ctrl_qubits + 1], fc=ec, ec=ec, tc=tc)

            self._sidetext(node, qubit_b, tc=tc, text=f"{gate_text} ({self._data[node]['param']})")
            self._line(qubit_b, qubit_t, lc=lc)

    def _swap(self, xy, node, color=None):
        """Draw a Swap gate"""
        self._swap_cross(xy[0], color=color)
        self._swap_cross(xy[1], color=color)
        self._line(xy[0], xy[1], lc=color)

        # add calibration text
        gate_text = self._data[node]["gate_text"].split("\n")[-1]
        if self._data[node]["raw_gate_text"] in self._calibrations:
            xpos, ypos = xy[0]
            self._ax.text(
                xpos,
                ypos + 0.7 * HIG,
                gate_text,
                ha="center",
                va="top",
                fontsize=self._style["sfs"],
                color=self._style["tc"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _swap_cross(self, xy, color=None):
        """Draw the Swap cross symbol"""
        xpos, ypos = xy

        self._ax.plot(
            [xpos - 0.20 * WID, xpos + 0.20 * WID],
            [ypos - 0.20 * WID, ypos + 0.20 * WID],
            color=color,
            linewidth=self._lwidth2,
            zorder=PORDER_LINE + 1,
        )
        self._ax.plot(
            [xpos - 0.20 * WID, xpos + 0.20 * WID],
            [ypos + 0.20 * WID, ypos - 0.20 * WID],
            color=color,
            linewidth=self._lwidth2,
            zorder=PORDER_LINE + 1,
        )

    def _sidetext(self, node, xy, tc=None, text=""):
        """Draw the sidetext for symmetric gates"""
        xpos, ypos = xy

        # 0.11 = the initial gap, add 1/2 text width to place on the right
        xp = xpos + 0.11 + self._data[node]["width"] / 2
        self._ax.text(
            xp,
            ypos + HIG,
            text,
            ha="center",
            va="top",
            fontsize=self._sfs,
            color=tc,
            clip_on=True,
            zorder=PORDER_TEXT,
        )

    def _line(self, xy0, xy1, lc=None, ls=None, zorder=PORDER_LINE):
        """Draw a line from xy0 to xy1"""
        x0, y0 = xy0
        x1, y1 = xy1
        linecolor = self._style["lc"] if lc is None else lc
        linestyle = "solid" if ls is None else ls

        if linestyle == "doublet":
            theta = np.arctan2(np.abs(x1 - x0), np.abs(y1 - y0))
            dx = 0.05 * WID * np.cos(theta)
            dy = 0.05 * WID * np.sin(theta)
            self._ax.plot(
                [x0 + dx, x1 + dx],
                [y0 + dy, y1 + dy],
                color=linecolor,
                linewidth=self._lwidth2,
                linestyle="solid",
                zorder=zorder,
            )
            self._ax.plot(
                [x0 - dx, x1 - dx],
                [y0 - dy, y1 - dy],
                color=linecolor,
                linewidth=self._lwidth2,
                linestyle="solid",
                zorder=zorder,
            )
        else:
            self._ax.plot(
                [x0, x1],
                [y0, y1],
                color=linecolor,
                linewidth=self._lwidth2,
                linestyle=linestyle,
                zorder=zorder,
            )


class Anchor:
    """Locate the anchors for the gates"""

    def __init__(self, reg_num, yind, fold):
        self._yind = yind
        self._fold = fold
        self._reg_num = reg_num
        self._gate_placed = []
        self.nxt_anchor_idx = 0
        self.gate_anchor = 0

    def plot_coord(self, index, gate_width, x_offset):
        """Set the coord positions for an index"""
        h_pos = index % self._fold + 1
        # check folding
        if self._fold > 0:
            if h_pos + (gate_width - 1) > self._fold:
                index += self._fold - (h_pos - 1)
            x_pos = index % self._fold + 0.5 * gate_width + 0.04
            y_pos = self._yind - (index // self._fold) * (self._reg_num + 1)
        else:
            x_pos = index + 0.5 * gate_width + 0.04
            y_pos = self._yind

        # could have been updated, so need to store
        self.gate_anchor = index
        return x_pos + x_offset, y_pos

    def set_index(self, index, layer_width):
        """Set the index for a gate"""
        if self._fold < 2:
            _index = index
        else:
            h_pos = index % self._fold + 1
            if h_pos + (layer_width - 1) > self._fold:
                _index = index + self._fold - (h_pos - 1) + 1
            else:
                _index = index
        for ii in range(layer_width):
            idx = _index + ii
            if idx not in self._gate_placed:
                self._gate_placed.append(idx)
                self.nxt_anchor_idx = idx + 1

    def get_index(self):
        """Getter for the index"""
        if self._gate_placed:
            return self._gate_placed[-1] + 1
        return 0


class HasMatplotlibWrapper:
    """Wrapper to lazily import matplotlib."""

    has_matplotlib = False

    # pylint: disable=unused-import
    def __bool__(self):
        if not self.has_matplotlib:
            try:
                from matplotlib import get_backend
                from matplotlib import patches
                from matplotlib import pyplot as plt

                self.has_matplotlib = True
            except ImportError:
                self.has_matplotlib = False
        return self.has_matplotlib


HAS_MATPLOTLIB = HasMatplotlibWrapper()
