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

import collections
import itertools
import re
from warnings import warn

import numpy as np


try:
    from pylatexenc.latex2text import LatexNodes2Text

    HAS_PYLATEX = True
except ImportError:
    HAS_PYLATEX = False

from qiskit.circuit import ControlledGate
from qiskit.visualization.qcstyle import load_style
from qiskit.visualization.utils import get_gate_ctrl_text, get_param_str
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.circuit.tools.pi_check import pi_check

# Default gate width and height
WID = 0.65
HIG = 0.65

BASE_SIZE = 3.01
PORDER_GATE = 5
PORDER_LINE = 3
PORDER_REGLINE = 2
PORDER_GRAY = 3
PORDER_TEXT = 6
PORDER_SUBP = 4


class Anchor:
    """Locate the anchors for the gates"""

    def __init__(self, reg_num, yind, fold):
        self.__yind = yind
        self.__fold = fold
        self.__reg_num = reg_num
        self.__gate_placed = []
        self.gate_anchor = 0

    def plot_coord(self, index, gate_width, x_offset):
        """Set the coord positions for an index"""
        h_pos = index % self.__fold + 1
        # check folding
        if self.__fold > 0:
            if h_pos + (gate_width - 1) > self.__fold:
                index += self.__fold - (h_pos - 1)
            x_pos = index % self.__fold + 0.5 * gate_width + 0.04
            y_pos = self.__yind - (index // self.__fold) * (self.__reg_num + 1)
        else:
            x_pos = index + 0.5 * gate_width + 0.04
            y_pos = self.__yind

        # could have been updated, so need to store
        self.gate_anchor = index
        return x_pos + x_offset, y_pos

    def is_locatable(self, index, gate_width):
        """Determine if a gate has been placed"""
        hold = [index + i for i in range(gate_width)]
        for p in hold:
            if p in self.__gate_placed:
                return False
        return True

    def set_index(self, index, gate_width):
        """Set the index for a gate"""
        if self.__fold < 2:
            _index = index
        else:
            h_pos = index % self.__fold + 1
            if h_pos + (gate_width - 1) > self.__fold:
                _index = index + self.__fold - (h_pos - 1) + 1
            else:
                _index = index
        for ii in range(gate_width):
            if _index + ii not in self.__gate_placed:
                self.__gate_placed.append(_index + ii)
        self.__gate_placed.sort()

    def get_index(self):
        """Getter for the index"""
        if self.__gate_placed:
            return self.__gate_placed[-1] + 1
        return 0


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
    ):

        if not HAS_MATPLOTLIB:
            raise MissingOptionalLibraryError(
                libname="Matplotlib",
                name="MatplotlibDrawer",
                pip_install="pip install matplotlib",
            )
        from matplotlib import patches

        self.patches_mod = patches
        from matplotlib import pyplot as plt

        self.plt_mod = plt
        if not HAS_PYLATEX:
            raise MissingOptionalLibraryError(
                libname="pylatexenc",
                name="MatplotlibDrawer",
                pip_install="pip install pylatexenc",
            )
        self._clbit = []
        self._qubit = []
        self._registers(clbits, qubits)
        self._bit_locations = {
            bit: {"register": register, "index": index}
            for register in cregs + qregs
            for index, bit in enumerate(register)
        }
        for index, bit in list(enumerate(qubits)) + list(enumerate(clbits)):
            if bit not in self._bit_locations:
                self._bit_locations[bit] = {"register": None, "index": index}

        self._qubit_dict = collections.OrderedDict()
        self._clbit_dict = collections.OrderedDict()
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
            self._return_fig = True
            self._figure = plt.figure()
            self._figure.patch.set_facecolor(color=self._style["bg"])
            self._ax = self._figure.add_subplot(111)
        else:
            self._return_fig = False
            self._ax = ax
            self._figure = ax.get_figure()
        self._ax.axis("off")
        self._ax.set_aspect("equal")
        self._ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        self._initial_state = initial_state
        self._cregbundle = cregbundle
        self._set_cregbundle()
        self._global_phase = global_phase

        self._n_lines = 0
        self._xmax = 0
        self._ymax = 0
        self._x_offset = 0
        self._reg_long_text = 0
        self._style["fs"] *= self._scale
        self._style["sfs"] *= self._scale
        self._lwidth15 = 1.5 * self._scale
        self._lwidth2 = 2.0 * self._scale
        self._gate_width = {}

        # these char arrays are for finding text_width when not
        # using get_renderer method for the matplotlib backend
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

    def _registers(self, clbit, qubit):
        self._clbit = []
        for r in clbit:
            self._clbit.append(r)
        self._qubit = []
        for r in qubit:
            self._qubit.append(r)

    def _set_cregbundle(self):
        """Sets the cregbundle to False if there is any instruction that
        needs access to individual clbit."""
        for layer in self._nodes:
            for node in layer:
                if node.cargs and node.op.name != "measure":
                    self._cregbundle = False
                    warn(
                        "Cregbundle set to False since an instruction needs to refer"
                        " to individual classical wire",
                        RuntimeWarning,
                        2,
                    )
                    break
            else:
                continue
            break

    # This computes the width of a string in the default font
    def _get_text_width(self, text, fontsize, param=False):
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

        f = 0 if fontsize == self._style["fs"] else 1
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

    def _get_colors(self, op, gate_text):
        base_name = None if not hasattr(op, "base_gate") else op.base_gate.name
        color = None
        if gate_text in self._style["dispcol"]:
            color = self._style["dispcol"][gate_text]
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
        return fc, ec, gt, self._style["tc"], sc, lc

    def _multiqubit_gate(
        self, node, xy, c_xy=None, fc=None, ec=None, gt=None, sc=None, text="", subtext=""
    ):
        xpos = min(x[0] for x in xy)
        ypos = min(y[1] for y in xy)
        ypos_max = max(y[1] for y in xy)
        if c_xy:
            cxpos = min(x[0] for x in c_xy)
            cypos = min(y[1] for y in c_xy)
            ypos = min(ypos, cypos)
        fs = self._style["fs"]
        sfs = self._style["sfs"]

        wid = max(self._gate_width[node] + 0.21, WID)

        qubit_span = abs(ypos) - abs(ypos_max) + 1
        height = HIG + (qubit_span - 1)
        box = self.patches_mod.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG),
            width=wid,
            height=height,
            fc=fc,
            ec=ec,
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
                fontsize=fs,
                color=gt,
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
                    fontsize=fs,
                    color=gt,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
        if text:
            if subtext:
                self._ax.text(
                    xpos + 0.11,
                    ypos + 0.4 * height,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=gt,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
                self._ax.text(
                    xpos + 0.11,
                    ypos + 0.2 * height,
                    subtext,
                    ha="center",
                    va="center",
                    fontsize=sfs,
                    color=sc,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
            else:
                self._ax.text(
                    xpos + 0.11,
                    ypos + 0.5 * (qubit_span - 1),
                    text,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=gt,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                    wrap=True,
                )

    def _gate(self, node, xy, fc=None, ec=None, gt=None, sc=None, text="", subtext=""):
        xpos, ypos = xy
        fs = self._style["fs"]
        sfs = self._style["sfs"]

        wid = max(self._gate_width[node], WID)

        box = self.patches_mod.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG),
            width=wid,
            height=HIG,
            fc=fc,
            ec=ec,
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        if text:
            if subtext:
                self._ax.text(
                    xpos,
                    ypos + 0.15 * HIG,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=gt,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
                self._ax.text(
                    xpos,
                    ypos - 0.3 * HIG,
                    subtext,
                    ha="center",
                    va="center",
                    fontsize=sfs,
                    color=sc,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
            else:
                self._ax.text(
                    xpos,
                    ypos,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=gt,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )

    def _sidetext(self, node, xy, tc=None, text=""):
        xpos, ypos = xy

        # 0.11 = the initial gap, add 1/2 text width to place on the right
        text_width = self._gate_width[node]
        xp = xpos + 0.11 + text_width / 2
        self._ax.text(
            xp,
            ypos + HIG,
            text,
            ha="center",
            va="top",
            fontsize=self._style["sfs"],
            color=tc,
            clip_on=True,
            zorder=PORDER_TEXT,
        )

    def _line(self, xy0, xy1, lc=None, ls=None, zorder=PORDER_LINE):
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

    def _measure(self, node, qxy, cxy, cid, fc=None, ec=None, gt=None, sc=None):
        qx, qy = qxy
        cx, cy = cxy

        # draw gate box
        self._gate(node, qxy, fc=fc, ec=ec, gt=gt, sc=sc)

        # add measure symbol
        arc = self.patches_mod.Arc(
            xy=(qx, qy - 0.15 * HIG),
            width=WID * 0.7,
            height=HIG * 0.7,
            theta1=0,
            theta2=180,
            fill=False,
            ec=gt,
            linewidth=self._lwidth2,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(arc)
        self._ax.plot(
            [qx, qx + 0.35 * WID],
            [qy - 0.15 * HIG, qy + 0.20 * HIG],
            color=gt,
            linewidth=self._lwidth2,
            zorder=PORDER_GATE,
        )
        # arrow
        self._line(qxy, [cx, cy + 0.35 * WID], lc=self._style["cc"], ls=self._style["cline"])
        arrowhead = self.patches_mod.Polygon(
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
                fontsize=0.8 * self._style["fs"],
                color=self._style["tc"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _conditional(self, xy, istrue=False):
        xpos, ypos = xy

        fc = self._style["lc"] if istrue else self._style["bg"]
        box = self.patches_mod.Circle(
            xy=(xpos, ypos),
            radius=WID * 0.15,
            fc=fc,
            ec=self._style["lc"],
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

    def _ctrl_qubit(self, xy, fc=None, ec=None, tc=None, text="", text_top=None):
        xpos, ypos = xy
        box = self.patches_mod.Circle(
            xy=(xpos, ypos),
            radius=WID * 0.15,
            fc=fc,
            ec=ec,
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)
        # display the control label at the top or bottom if there is one
        if text_top is True:
            self._ax.text(
                xpos,
                ypos + 0.7 * HIG,
                text,
                ha="center",
                va="top",
                fontsize=self._style["sfs"],
                color=tc,
                clip_on=True,
                zorder=PORDER_TEXT,
            )
        elif text_top is False:
            self._ax.text(
                xpos,
                ypos - 0.3 * HIG,
                text,
                ha="center",
                va="top",
                fontsize=self._style["sfs"],
                color=tc,
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _set_ctrl_bits(
        self, ctrl_state, num_ctrl_qubits, qbit, ec=None, tc=None, text="", qargs=None
    ):
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

    def _x_tgt_qubit(self, xy, ec=None, ac=None):
        linewidth = self._lwidth2
        xpos, ypos = xy
        box = self.patches_mod.Circle(
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

    def _swap(self, xy, color=None):
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

    def _barrier(self, config):
        xys = config["coord"]
        for xy in xys:
            xpos, ypos = xy
            self._ax.plot(
                [xpos, xpos],
                [ypos + 0.5, ypos - 0.5],
                linewidth=self._scale,
                linestyle="dashed",
                color=self._style["lc"],
                zorder=PORDER_TEXT,
            )
            box = self.patches_mod.Rectangle(
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

    def draw(self, filename=None, verbose=False):
        """Draw method called from circuit_drawer"""
        self._draw_regs()
        self._draw_ops(verbose)
        _xl = -self._style["margin"][0]
        _xr = self._xmax + self._style["margin"][1]
        _yb = -self._ymax - self._style["margin"][2] + 1 - 0.5
        _yt = self._style["margin"][3] + 0.5
        self._ax.set_xlim(_xl, _xr)
        self._ax.set_ylim(_yb, _yt)

        # update figure size
        fig_w = _xr - _xl
        fig_h = _yt - _yb
        if self._style["figwidth"] < 0.0:
            self._style["figwidth"] = fig_w * BASE_SIZE * self._style["fs"] / 72 / WID
        self._figure.set_size_inches(
            self._style["figwidth"], self._style["figwidth"] * fig_h / fig_w
        )
        if self._global_phase:
            self.plt_mod.text(
                _xl, _yt, "Global Phase: %s" % pi_check(self._global_phase, output="mpl")
            )

        if filename:
            self._figure.savefig(
                filename,
                dpi=self._style["dpi"],
                bbox_inches="tight",
                facecolor=self._figure.get_facecolor(),
            )
        if self._return_fig:
            from matplotlib import get_backend

            if get_backend() in ["module://ipykernel.pylab.backend_inline", "nbAgg"]:
                self.plt_mod.close(self._figure)
            return self._figure

    def _draw_regs(self):
        longest_reg_name_width = 0
        initial_qbit = " |0>" if self._initial_state else ""
        initial_cbit = " 0" if self._initial_state else ""

        def _fix_double_script(reg_name):
            words = reg_name.split(" ")
            words = [word.replace("_", r"\_") if word.count("_") > 1 else word for word in words]
            words = [
                word.replace("^", r"\^{\ }") if word.count("^") > 1 else word for word in words
            ]
            reg_name = " ".join(words).replace(" ", "\\;")
            return reg_name

        # quantum register
        fs = self._style["fs"]
        for ii, reg in enumerate(self._qubit):
            register = self._bit_locations[reg]["register"]
            index = self._bit_locations[reg]["index"]

            if len(self._qubit) > 1:
                if self._layout is None:
                    qubit_name = f"${{{register.name}}}_{{{index}}}$"
                else:
                    if self._layout[index]:
                        virt_bit = self._layout[index]
                        try:
                            virt_reg = next(
                                reg for reg in self._layout.get_registers() if virt_bit in reg
                            )
                            qubit_name = "${{{name}}}_{{{index}}} \\mapsto {{{physical}}}$".format(
                                name=virt_reg.name,
                                index=virt_reg[:].index(virt_bit),
                                physical=index,
                            )

                        except StopIteration:
                            qubit_name = "${{{name}}} \\mapsto {{{physical}}}$".format(
                                name=virt_bit, physical=index
                            )
                    else:
                        qubit_name = f"${{{index}}}$"
            else:
                qubit_name = f"{register.name}"
            qubit_name = _fix_double_script(qubit_name) + initial_qbit
            text_width = self._get_text_width(qubit_name, fs) * 1.15

            if text_width > longest_reg_name_width:
                longest_reg_name_width = text_width
            pos = -ii
            self._qubit_dict[ii] = {
                "y": pos,
                "reg_name": qubit_name,
                "index": index,
                "group": register,
            }
            self._n_lines += 1

        # classical register
        if self._clbit:
            n_clbit = self._clbit.copy()
            n_clbit.pop(0)
            idx = 0
            y_off = -len(self._qubit)
            for ii, (reg, nreg) in enumerate(itertools.zip_longest(self._clbit, n_clbit)):
                pos = y_off - idx
                register = self._bit_locations[reg]["register"]
                index = self._bit_locations[reg]["index"]

                if self._cregbundle:
                    clbit_name = f"{register.name}"
                    clbit_name = _fix_double_script(clbit_name) + initial_cbit
                    text_width = self._get_text_width(register.name, fs) * 1.15
                    if text_width > longest_reg_name_width:
                        longest_reg_name_width = text_width
                    self._clbit_dict[ii] = {
                        "y": pos,
                        "reg_name": clbit_name,
                        "index": index,
                        "group": register,
                    }
                    if not (not nreg or register != self._bit_locations[nreg]["register"]):
                        continue
                else:
                    clbit_name = f"${register.name}_{{{index}}}$"
                    clbit_name = _fix_double_script(clbit_name) + initial_cbit
                    text_width = self._get_text_width(register.name, fs) * 1.15
                    if text_width > longest_reg_name_width:
                        longest_reg_name_width = text_width
                    self._clbit_dict[ii] = {
                        "y": pos,
                        "reg_name": clbit_name,
                        "index": index,
                        "group": register,
                    }
                self._n_lines += 1
                idx += 1

        self._reg_long_text = longest_reg_name_width
        self._x_offset = -1.2 + self._reg_long_text

    def _draw_regs_sub(self, n_fold, feedline_l=False, feedline_r=False):
        # quantum register
        fs = self._style["fs"]
        for qubit in self._qubit_dict.values():
            qubit_name = qubit["reg_name"]
            y = qubit["y"] - n_fold * (self._n_lines + 1)
            self._ax.text(
                self._x_offset - 0.2,
                y,
                qubit_name,
                ha="right",
                va="center",
                fontsize=1.25 * fs,
                color=self._style["tc"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )
            self._line([self._x_offset, y], [self._xmax, y], zorder=PORDER_REGLINE)

        # classical register
        this_clbit_dict = {}
        for clbit in self._clbit_dict.values():
            clbit_name = clbit["reg_name"]
            y = clbit["y"] - n_fold * (self._n_lines + 1)
            if y not in this_clbit_dict.keys():
                this_clbit_dict[y] = {"val": 1, "reg_name": clbit_name}
            else:
                this_clbit_dict[y]["val"] += 1
        for y, this_clbit in this_clbit_dict.items():
            # cregbundle
            if this_clbit["val"] > 1:
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
                    fontsize=0.8 * fs,
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
            self._ax.text(
                self._x_offset - 0.2,
                y,
                this_clbit["reg_name"],
                ha="right",
                va="center",
                fontsize=1.25 * fs,
                color=self._style["tc"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )
            self._line(
                [self._x_offset, y],
                [self._xmax, y],
                lc=self._style["cc"],
                ls=self._style["cline"],
                zorder=PORDER_REGLINE,
            )

        # lf vertical line at either end
        if feedline_l or feedline_r:
            xpos_l = self._x_offset - 0.01
            xpos_r = self._fold + self._x_offset + 0.1
            ypos1 = -n_fold * (self._n_lines + 1)
            ypos2 = -(n_fold + 1) * (self._n_lines) - n_fold + 1
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

    def _draw_ops(self, verbose=False):
        _standard_1q_gates = [
            "x",
            "y",
            "z",
            "id",
            "h",
            "r",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
            "rxx",
            "ryy",
            "rzx",
            "u1",
            "u2",
            "u3",
            "u",
            "swap",
            "reset",
            "sx",
            "sxdg",
            "p",
        ]

        # generate coordinate manager
        q_anchors = {}
        for key, qubit in self._qubit_dict.items():
            q_anchors[key] = Anchor(reg_num=self._n_lines, yind=qubit["y"], fold=self._fold)
        c_anchors = {}
        for key, clbit in self._clbit_dict.items():
            c_anchors[key] = Anchor(reg_num=self._n_lines, yind=clbit["y"], fold=self._fold)
        #
        # draw the ops
        #
        prev_anc = -1
        fs = self._style["fs"]
        sfs = self._style["sfs"]
        for layer in self._nodes:
            widest_box = 0.0
            self._gate_width = {}
            #
            # compute the layer_width for this layer
            #
            for node in layer:
                op = node.op
                self._gate_width[node] = WID

                if op._directive or op.name == "measure":
                    continue

                base_name = None if not hasattr(op, "base_gate") else op.base_gate.name
                gate_text, ctrl_text, _ = get_gate_ctrl_text(op, "mpl", style=self._style)

                # if a standard_gate, no params, and no labels, layer_width is 1
                if not hasattr(op, "params") and (
                    (op.name in _standard_1q_gates or base_name in _standard_1q_gates)
                    and gate_text in (op.name, base_name)
                    and ctrl_text is None
                ):
                    continue

                # small increments at end of the 3 _get_text_width calls are for small
                # spacing adjustments between gates
                ctrl_width = self._get_text_width(ctrl_text, fontsize=sfs) - 0.05

                # get param_width, but 0 for gates with array params
                if (
                    hasattr(op, "params")
                    and not any(isinstance(param, np.ndarray) for param in op.params)
                    and len(op.params) > 0
                ):
                    param = get_param_str(op, "mpl", ndigits=3)
                    if op.name == "initialize":
                        param = "[%s]" % param
                    raw_param_width = self._get_text_width(param, fontsize=sfs, param=True)
                    param_width = raw_param_width + 0.08
                else:
                    param_width = raw_param_width = 0.0

                if op.name == "rzz" or base_name in ["u1", "p", "rzz"]:
                    if base_name == "u1":
                        tname = "U1"
                    elif base_name == "p":
                        tname = "P"
                    else:
                        tname = "ZZ"
                    raw_gate_width = (
                        self._get_text_width(tname + " ()", fontsize=sfs) + raw_param_width
                    )
                    gate_width = (raw_gate_width + 0.08) * 1.5
                else:
                    raw_gate_width = self._get_text_width(gate_text, fontsize=fs)
                    gate_width = raw_gate_width + 0.10
                    # add .21 for the qubit numbers on the left of the multibit gates
                    if op.name not in _standard_1q_gates and base_name not in _standard_1q_gates:
                        gate_width += 0.21

                box_width = max(gate_width, ctrl_width, param_width, WID)
                if box_width > widest_box:
                    widest_box = box_width
                self._gate_width[node] = max(raw_gate_width, raw_param_width)

            layer_width = int(widest_box) + 1
            this_anc = prev_anc + 1
            #
            # draw the gates in this layer
            #
            for node in layer:
                op = node.op
                base_name = None if not hasattr(op, "base_gate") else op.base_gate.name
                gate_text, ctrl_text, raw_gate_text = get_gate_ctrl_text(
                    op, "mpl", style=self._style
                )
                fc, ec, gt, tc, sc, lc = self._get_colors(op, raw_gate_text)

                # get qubit index
                q_idxs = []
                for qarg in node.qargs:
                    for index, reg in self._qubit_dict.items():
                        if (
                            reg["group"] == self._bit_locations[qarg]["register"]
                            and reg["index"] == self._bit_locations[qarg]["index"]
                        ):
                            q_idxs.append(index)
                            break

                # get clbit index
                c_idxs = []
                for carg in node.cargs:
                    for index, reg in self._clbit_dict.items():
                        if (
                            reg["group"] == self._bit_locations[carg]["register"]
                            and reg["index"] == self._bit_locations[carg]["index"]
                        ):
                            c_idxs.append(index)
                            break

                # only add the gate to the anchors if it is going to be plotted.
                # this prevents additional blank wires at the end of the line if
                # the last instruction is a barrier type
                if self._plot_barriers or not op._directive:
                    for ii in q_idxs:
                        q_anchors[ii].set_index(this_anc, layer_width)

                # qubit coordinate
                q_xy = [
                    q_anchors[ii].plot_coord(this_anc, layer_width, self._x_offset) for ii in q_idxs
                ]
                # clbit coordinate
                c_xy = [
                    c_anchors[ii].plot_coord(this_anc, layer_width, self._x_offset) for ii in c_idxs
                ]
                # bottom and top point of qubit
                qubit_b = min(q_xy, key=lambda xy: xy[1])
                qubit_t = max(q_xy, key=lambda xy: xy[1])

                # update index based on the value from plotting
                this_anc = q_anchors[q_idxs[0]].gate_anchor

                if verbose:
                    print(op)

                # load param
                if (
                    hasattr(op, "params")
                    and len(op.params) > 0
                    and not any(isinstance(param, np.ndarray) for param in op.params)
                ):
                    param = f"{get_param_str(op, 'mpl', ndigits=3)}"
                else:
                    param = ""

                # conditional gate
                if op.condition:
                    c_xy = [
                        c_anchors[ii].plot_coord(this_anc, layer_width, self._x_offset)
                        for ii in self._clbit_dict
                    ]
                    mask = 0
                    for index, cbit in enumerate(self._clbit):
                        if self._bit_locations[cbit]["register"] == op.condition[0]:
                            mask |= 1 << index
                    val = op.condition[1]
                    # cbit list to consider
                    fmt_c = f"{{:0{len(c_xy)}b}}"
                    cmask = list(fmt_c.format(mask))[::-1]
                    # value
                    fmt_v = f"{{:0{cmask.count('1')}b}}"
                    vlist = list(fmt_v.format(val))
                    if not self._reverse_bits:
                        vlist = vlist[::-1]

                    # plot conditionals
                    v_ind = 0
                    xy_plot = []
                    for xy, m in zip(c_xy, cmask):
                        if m == "1":
                            if xy not in xy_plot:
                                if vlist[v_ind] == "1" or self._cregbundle:
                                    self._conditional(xy, istrue=True)
                                else:
                                    self._conditional(xy, istrue=False)
                                xy_plot.append(xy)
                            v_ind += 1
                    clbit_b = sorted(xy_plot, key=lambda xy: xy[1])[0]
                    xpos, ypos = clbit_b
                    self._ax.text(
                        xpos,
                        ypos - 0.3 * HIG,
                        hex(val),
                        ha="center",
                        va="top",
                        fontsize=sfs,
                        color=self._style["tc"],
                        clip_on=True,
                        zorder=PORDER_TEXT,
                    )
                    self._line(qubit_t, clbit_b, lc=self._style["cc"], ls=self._style["cline"])
                #
                # draw special gates
                #
                if op.name == "measure":
                    vv = self._clbit_dict[c_idxs[0]]["index"]
                    self._measure(node, q_xy[0], c_xy[0], vv, fc=fc, ec=ec, gt=gt, sc=sc)

                elif op._directive:
                    _barriers = {"coord": [], "group": []}
                    for index, qbit in enumerate(q_idxs):
                        q_group = self._qubit_dict[qbit]["group"]
                        if q_group not in _barriers["group"]:
                            _barriers["group"].append(q_group)
                        _barriers["coord"].append(q_xy[index])
                    if self._plot_barriers:
                        self._barrier(_barriers)

                elif op.name == "initialize":
                    vec = f"$[{param.replace('$', '')}]$"
                    if len(q_xy) == 1:
                        self._gate(
                            node, q_xy[0], fc=fc, ec=ec, gt=gt, sc=sc, text=gate_text, subtext=vec
                        )
                    else:
                        self._multiqubit_gate(
                            node, q_xy, fc=fc, ec=ec, gt=gt, sc=sc, text=gate_text, subtext=vec
                        )
                #
                # draw single qubit gates
                #
                elif len(q_xy) == 1 and not node.cargs:
                    self._gate(
                        node,
                        q_xy[0],
                        fc=fc,
                        ec=ec,
                        gt=gt,
                        sc=sc,
                        text=gate_text,
                        subtext=str(param),
                    )
                #
                # draw controlled and special gates
                #
                # cz and mcz gates
                elif op.name != "z" and base_name == "z":
                    num_ctrl_qubits = op.num_ctrl_qubits
                    self._set_ctrl_bits(
                        op.ctrl_state,
                        num_ctrl_qubits,
                        q_xy,
                        ec=ec,
                        tc=tc,
                        text=ctrl_text,
                        qargs=node.qargs,
                    )
                    self._ctrl_qubit(q_xy[-1], fc=ec, ec=ec, tc=tc)
                    self._line(qubit_b, qubit_t, lc=lc, zorder=PORDER_LINE + 1)

                # cu1, cp, rzz, and controlled rzz gates (sidetext gates)
                elif op.name == "rzz" or base_name in ["u1", "p", "rzz"]:
                    num_ctrl_qubits = 0 if op.name == "rzz" else op.num_ctrl_qubits
                    if op.name != "rzz":
                        self._set_ctrl_bits(
                            op.ctrl_state,
                            num_ctrl_qubits,
                            q_xy,
                            ec=ec,
                            tc=tc,
                            text=ctrl_text,
                            qargs=node.qargs,
                        )
                    self._ctrl_qubit(q_xy[num_ctrl_qubits], fc=ec, ec=ec, tc=tc)
                    if base_name not in ["u1", "p"]:
                        self._ctrl_qubit(q_xy[num_ctrl_qubits + 1], fc=ec, ec=ec, tc=tc)
                    if base_name == "u1":
                        if self._style["disptex"]["u1"].find("\\mathrm") >= 0:
                            stext = self._style["disptex"]["u1"]
                        else:
                            stext = f"$\\mathrm{{{self._style['disptex']['u1']}}}$"
                    elif base_name == "p":
                        stext = "P"
                    else:
                        stext = "ZZ"
                    self._sidetext(node, qubit_b, tc=tc, text=f"{stext} ({param})")
                    self._line(qubit_b, qubit_t, lc=lc)

                # swap gate
                elif op.name == "swap":
                    self._swap(q_xy[0], color=lc)
                    self._swap(q_xy[1], color=lc)
                    self._line(qubit_b, qubit_t, lc=lc)

                # cswap gate
                elif op.name != "swap" and base_name == "swap":
                    num_ctrl_qubits = op.num_ctrl_qubits
                    self._set_ctrl_bits(
                        op.ctrl_state,
                        num_ctrl_qubits,
                        q_xy,
                        ec=ec,
                        tc=tc,
                        text=ctrl_text,
                        qargs=node.qargs,
                    )
                    self._swap(q_xy[num_ctrl_qubits], color=lc)
                    self._swap(q_xy[num_ctrl_qubits + 1], color=lc)
                    self._line(qubit_b, qubit_t, lc=lc)

                # all other controlled gates
                elif isinstance(op, ControlledGate):
                    num_ctrl_qubits = op.num_ctrl_qubits
                    num_qargs = len(q_xy) - num_ctrl_qubits
                    self._set_ctrl_bits(
                        op.ctrl_state,
                        num_ctrl_qubits,
                        q_xy,
                        ec=ec,
                        tc=tc,
                        text=ctrl_text,
                        qargs=node.qargs,
                    )
                    self._line(qubit_b, qubit_t, lc=lc)
                    if num_qargs == 1 and base_name == "x":
                        tgt_color = self._style["dispcol"]["target"]
                        tgt = tgt_color if isinstance(tgt_color, str) else tgt_color[0]
                        self._x_tgt_qubit(q_xy[num_ctrl_qubits], ec=ec, ac=tgt)
                    elif num_qargs == 1:
                        self._gate(
                            node,
                            q_xy[num_ctrl_qubits],
                            fc=fc,
                            ec=ec,
                            gt=gt,
                            sc=sc,
                            text=gate_text,
                            subtext=f"{param}",
                        )
                    else:
                        self._multiqubit_gate(
                            node,
                            q_xy[num_ctrl_qubits:],
                            fc=fc,
                            ec=ec,
                            gt=gt,
                            sc=sc,
                            text=gate_text,
                            subtext=f"{param}",
                        )

                # draw multi-qubit gate as final default
                else:
                    self._multiqubit_gate(
                        node,
                        q_xy,
                        c_xy,
                        fc=fc,
                        ec=ec,
                        gt=gt,
                        sc=sc,
                        text=gate_text,
                        subtext=f"{param}",
                    )

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self._plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = -1 if all(op._directive for node in layer) else 0

            prev_anc = this_anc + layer_width + barrier_offset - 1
        #
        # adjust window size and draw horizontal lines
        #
        anchors = [q_anchors[ii].get_index() for ii in self._qubit_dict]
        max_anc = max(anchors) if anchors else 0
        n_fold = max(0, max_anc - 1) // self._fold if self._fold > 0 else 0

        # window size
        if max_anc > self._fold > 0:
            self._xmax = self._fold + 1 + self._x_offset - 0.9
            self._ymax = (n_fold + 1) * (self._n_lines + 1) - 1
        else:
            x_incr = 0.4 if not self._nodes else 0.9
            self._xmax = max_anc + 1 + self._x_offset - x_incr
            self._ymax = self._n_lines

        # add horizontal lines
        for ii in range(n_fold + 1):
            feedline_r = n_fold > 0 and n_fold > ii
            feedline_l = ii > 0
            self._draw_regs_sub(ii, feedline_l, feedline_r)

        # draw anchor index number
        if self._style["index"]:
            for ii in range(max_anc):
                if self._fold > 0:
                    x_coord = ii % self._fold + self._reg_long_text - 0.67
                    y_coord = -(ii // self._fold) * (self._n_lines + 1) + 0.7
                else:
                    x_coord = ii + self._reg_long_text - 0.67
                    y_coord = 0.7
                self._ax.text(
                    x_coord,
                    y_coord,
                    str(ii + 1),
                    ha="center",
                    va="center",
                    fontsize=sfs,
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )


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
