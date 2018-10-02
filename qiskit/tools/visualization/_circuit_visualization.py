# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string,missing-docstring

"""
Two quantum circuit drawers based on:
    1. LaTeX
    2. Matplotlib
"""

import json
import logging
import operator
import os
import re
import subprocess
import tempfile
from collections import namedtuple, OrderedDict
from fractions import Fraction
from io import StringIO
from itertools import groupby, zip_longest
from math import fmod, isclose, ceil

import numpy as np
from PIL import Image, ImageChops
from matplotlib import get_backend as get_matplotlib_backend, \
    patches as patches, pyplot as plt

from qiskit._qiskiterror import QISKitError
from qiskit.wrapper import load_qasm_file
from qiskit.dagcircuit import DAGCircuit
from qiskit.tools.visualization._error import VisualizationError
from qiskit.transpiler import transpile

logger = logging.getLogger(__name__)


def plot_circuit(circuit,
                 basis="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                       "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap",
                 scale=0.7,
                 style=None):
    """Plot and show circuit (opens new window, cannot inline in Jupyter)
    Defaults to an overcomplete basis, in order to not alter gates.
    """
    im = circuit_drawer(circuit, basis=basis, scale=scale, style=style)
    if im:
        im.show()


def circuit_drawer(circuit,
                   basis="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                         "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap",
                   scale=0.7,
                   filename=None,
                   style=None):
    """Draw a quantum circuit, via 2 methods (try 1st, if unsuccessful, 2nd):

    1. latex: high-quality images, but heavy external software dependencies
    2. matplotlib: purely in Python with no external dependencies

    Defaults to an overcomplete basis, in order to not alter gates.

    Args:
        circuit (QuantumCircuit): the quantum circuit to draw
        basis (str): the basis to unroll to prior to drawing
        scale (float): scale of image to draw (shrink if < 1)
        filename (str): file path to save image to
        style (dict or str): dictionary of style or file name of style file

    Returns:
        PIL.Image: an in-memory representation of the circuit diagram
    """
    try:
        return latex_circuit_drawer(circuit, basis, scale, filename, style)
    except (OSError, subprocess.CalledProcessError):
        return matplotlib_circuit_drawer(circuit, basis, scale, filename, style)


# -----------------------------------------------------------------------------
# Plot style sheet option
# -----------------------------------------------------------------------------
class QCStyle:
    def __init__(self):
        self.tc = '#000000'
        self.sc = '#000000'
        self.lc = '#000000'
        self.cc = '#778899'
        self.gc = '#ffffff'
        self.gt = '#000000'
        self.bc = '#bdbdbd'
        self.bg = '#ffffff'
        self.fs = 13
        self.sfs = 8
        self.disptex = {
            'id': 'id',
            'u0': 'U_0',
            'u1': 'U_1',
            'u2': 'U_2',
            'u3': 'U_3',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'h': 'H',
            's': 'S',
            'sdg': 'S^\\dagger',
            't': 'T',
            'tdg': 'T^\\dagger',
            'rx': 'R_x',
            'ry': 'R_y',
            'rz': 'R_z',
            'reset': '\\left|0\\right\\rangle'
        }
        self.dispcol = {
            'id': '#ffffff',
            'u0': '#ffffff',
            'u1': '#ffffff',
            'u2': '#ffffff',
            'u3': '#ffffff',
            'x': '#ffffff',
            'y': '#ffffff',
            'z': '#ffffff',
            'h': '#ffffff',
            's': '#ffffff',
            'sdg': '#ffffff',
            't': '#ffffff',
            'tdg': '#ffffff',
            'rx': '#ffffff',
            'ry': '#ffffff',
            'rz': '#ffffff',
            'reset': '#ffffff',
            'target': '#ffffff',
            'meas': '#ffffff'
        }
        self.latexmode = True
        self.pimode = False
        self.fold = 20
        self.bundle = False
        self.barrier = False
        self.index = False
        self.compress = True
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.0, 0.0, 0.3]
        self.cline = 'doublet'
        self.reverse = False

    def set_style(self, dic):
        self.tc = dic.get('textcolor', self.tc)
        self.sc = dic.get('subtextcolor', self.sc)
        self.lc = dic.get('linecolor', self.lc)
        self.cc = dic.get('creglinecolor', self.cc)
        self.gt = dic.get('gatetextcolor', self.tc)
        self.gc = dic.get('gatefacecolor', self.gc)
        self.bc = dic.get('barrierfacecolor', self.bc)
        self.bg = dic.get('backgroundcolor', self.bg)
        self.fs = dic.get('fontsize', self.fs)
        self.sfs = dic.get('subfontsize', self.sfs)
        self.disptex = dic.get('displaytext', self.disptex)
        for key in self.dispcol.keys():
            self.dispcol[key] = self.gc
        self.dispcol = dic.get('displaycolor', self.dispcol)
        self.latexmode = dic.get('latexdrawerstyle', self.latexmode)
        self.pimode = dic.get('usepiformat', self.pimode)
        self.fold = dic.get('fold', self.fold)
        if self.fold < 2:
            self.fold = -1
        self.bundle = dic.get('cregbundle', self.bundle)
        self.barrier = dic.get('plotbarrier', self.barrier)
        self.index = dic.get('showindex', self.index)
        self.compress = dic.get('compress', self.compress)
        self.figwidth = dic.get('figwidth', self.figwidth)
        self.dpi = dic.get('dpi', self.dpi)
        self.margin = dic.get('margin', self.margin)
        self.cline = dic.get('creglinestyle', self.cline)
        self.reverse = dic.get('reversebits', self.reverse)


def qx_color_scheme():
    return {
        "comment": "Style file for matplotlib_circuit_drawer (IBM QX Composer style)",
        "textcolor": "#000000",
        "gatetextcolor": "#000000",
        "subtextcolor": "#000000",
        "linecolor": "#000000",
        "creglinecolor": "#b9b9b9",
        "gatefacecolor": "#ffffff",
        "barrierfacecolor": "#bdbdbd",
        "backgroundcolor": "#ffffff",
        "fold": 20,
        "fontsize": 13,
        "subfontsize": 8,
        "figwidth": -1,
        "dpi": 150,
        "displaytext": {
            "id": "id",
            "u0": "U_0",
            "u1": "U_1",
            "u2": "U_2",
            "u3": "U_3",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "h": "H",
            "s": "S",
            "sdg": "S^\\dagger",
            "t": "T",
            "tdg": "T^\\dagger",
            "rx": "R_x",
            "ry": "R_y",
            "rz": "R_z",
            "reset": "\\left|0\\right\\rangle"
        },
        "displaycolor": {
            "id": "#ffca64",
            "u0": "#f69458",
            "u1": "#f69458",
            "u2": "#f69458",
            "u3": "#f69458",
            "x": "#a6ce38",
            "y": "#a6ce38",
            "z": "#a6ce38",
            "h": "#00bff2",
            "s": "#00bff2",
            "sdg": "#00bff2",
            "t": "#ff6666",
            "tdg": "#ff6666",
            "rx": "#ffca64",
            "ry": "#ffca64",
            "rz": "#ffca64",
            "reset": "#d7ddda",
            "target": "#00bff2",
            "meas": "#f070aa"
        },
        "latexdrawerstyle": True,
        "usepiformat": False,
        "cregbundle": False,
        "plotbarrier": False,
        "showindex": False,
        "compress": True,
        "margin": [2.0, 0.0, 0.0, 0.3],
        "creglinestyle": "solid",
        "reversebits": False
    }


# -----------------------------------------------------------------------------
# latex_circuit_drawer
# -----------------------------------------------------------------------------
def latex_circuit_drawer(circuit,
                         basis="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                               "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap",
                         scale=0.7,
                         filename=None,
                         style=None):
    """Draw a quantum circuit based on latex (Qcircuit package)

    Requires version >=2.6.0 of the qcircuit LaTeX package.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        basis (str): comma separated list of gates
        scale (float): scaling factor
        filename (str): file path to save image to
        style (dict or str): dictionary of style or file name of style file

    Returns:
        PIL.Image: an in-memory representation of the circuit diagram

    Raises:
        OSError: usually indicates that ```pdflatex``` or ```pdftocairo``` is
                 missing.
        CalledProcessError: usually points errors during diagram creation.
    """
    tmpfilename = 'circuit'
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, tmpfilename + '.tex')
        generate_latex_source(circuit, filename=tmppath, basis=basis,
                              scale=scale, style=style)
        im = None
        try:

            subprocess.run(["pdflatex", "-halt-on-error",
                            "-output-directory={}".format(tmpdirname),
                            "{}".format(tmpfilename + '.tex')],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           check=True)
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                logger.warning('WARNING: Unable to compile latex. '
                               'Is `pdflatex` installed? '
                               'Skipping latex circuit drawing...')
            raise
        except subprocess.CalledProcessError as e:
            with open('latex_error.log', 'wb') as error_file:
                error_file.write(e.stdout)
            logger.warning('WARNING Unable to complile latex. '
                           'The output from the pdflatex command can '
                           'be found in latex_error.log')
            raise
        else:
            try:
                base = os.path.join(tmpdirname, tmpfilename)
                subprocess.run(["pdftocairo", "-singlefile", "-png", "-q",
                                base + '.pdf', base])
                im = Image.open(base + '.png')
                im = _trim(im)
                os.remove(base + '.png')
                if filename:
                    im.save(filename, 'PNG')
            except OSError as e:
                if e.errno == os.errno.ENOENT:
                    logger.warning('WARNING: Unable to convert pdf to image. '
                                   'Is `poppler` installed? '
                                   'Skipping circuit drawing...')
                raise
        return im


def _trim(im):
    """Trim image and remove white space
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    return im


def generate_latex_source(circuit, filename=None,
                          basis="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                          "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap",
                          scale=0.7, style=None):
    """Convert QuantumCircuit to LaTeX string.

    Args:
        circuit (QuantumCircuit): input circuit
        scale (float): image scaling
        filename (str): optional filename to write latex
        basis (str): optional comma-separated list of gate names
        style (dict or str): dictionary of style or file name of style file

    Returns:
        str: Latex string appropriate for writing to file.
    """
    dag_circuit = DAGCircuit.fromQuantumCircuit(circuit, expand_gates=False)
    json_circuit = transpile(dag_circuit, basis_gates=basis, format='json')
    qcimg = QCircuitImage(json_circuit, scale, style=style)
    latex = qcimg.latex()
    if filename:
        with open(filename, 'w') as latex_file:
            latex_file.write(latex)
    return latex


class QCircuitImage(object):
    """This class contains methods to create \\LaTeX circuit images.

    The class targets the \\LaTeX package Q-circuit
    (https://arxiv.org/pdf/quant-ph/0406003).

    Thanks to Eric Sabo for the initial implementation for QISKit.
    """
    def __init__(self, circuit, scale, style=None):
        """
        Args:
            circuit (dict): compiled_circuit from qobj
            scale (float): image scaling
            style (dict or str): dictionary of style or file name of style file
        """
        # style sheet
        self._style = QCStyle()
        if style:
            if isinstance(style, dict):
                self._style.set_style(style)
            elif isinstance(style, str):
                with open(style, 'r') as infile:
                    dic = json.load(infile)
                self._style.set_style(dic)

        # compiled qobj circuit
        self.circuit = circuit

        # image scaling
        self.scale = scale

        # Map of qregs to sizes
        self.qregs = {}

        # Map of cregs to sizes
        self.cregs = {}

        # List of qregs and cregs in order of appearance in code and image
        self.ordered_regs = []

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
        self.sum_row_heights = 0

        # em points of separation between circuit columns
        self.column_separation = 0.5

        # em points of separation between circuit row
        self.row_separation = 0.0

        # presence of "box" or "target" determines row spacing
        self.has_box = False
        self.has_target = False

        #################################
        self.header = self.circuit['header']
        self.qregs = OrderedDict(_get_register_specs(
            self.header['qubit_labels']))
        self.qubit_list = []
        for qr in self.qregs:
            for i in range(self.qregs[qr]):
                self.qubit_list.append((qr, i))
        self.cregs = OrderedDict()
        if 'clbit_labels' in self.header:
            for item in self.header['clbit_labels']:
                self.cregs[item[0]] = item[1]
        self.clbit_list = []
        cregs = self.cregs
        if self._style.reverse:
            self.orig_cregs = self.cregs
            cregs = reversed(self.cregs)
        for cr in cregs:
            for i in range(self.cregs[cr]):
                self.clbit_list.append((cr, i))
        self.ordered_regs = [(item[0], item[1]) for
                             item in self.header['qubit_labels']]
        if self._style.reverse:
            reg_size = []
            reg_labels = []
            new_ordered_regs = []
            for regs in self.ordered_regs:
                if regs[0] in reg_labels:
                    continue
                reg_labels.append(regs[0])
                reg_size.append(len(
                    [x for x in self.ordered_regs if x[0] == regs[0]]))
            index = 0
            for size in reg_size:
                new_index = index + size
                for i in range(new_index - 1, index - 1, -1):
                    new_ordered_regs.append(self.ordered_regs[i])
                index = new_index
            self.ordered_regs = new_ordered_regs

        if 'clbit_labels' in self.header:
            for clabel in self.header['clbit_labels']:
                if self._style.reverse:
                    for cind in reversed(range(clabel[1])):
                        self.ordered_regs.append((clabel[0], cind))
                else:
                    for cind in range(clabel[1]):
                        self.ordered_regs.append((clabel[0], cind))
        self.img_regs = {bit: ind for ind, bit in
                         enumerate(self.ordered_regs)}
        self.img_width = len(self.img_regs)
        self.wire_type = {}
        for key, value in self.ordered_regs:
            self.wire_type[(key, value)] = key in self.cregs.keys()

    def latex(self, aliases=None):
        """Return LaTeX string representation of circuit.

        This method uses the LaTeX Qconfig package to create a graphical
        representation of the circuit.

        Returns:
            string: for writing to a LaTeX file.
        """
        self._initialize_latex_array(aliases)
        self._build_latex_array(aliases)
        header_1 = r"""% \documentclass[preview]{standalone}
% If the image is too large to fit on this documentclass use
\documentclass[draft]{beamer}
"""
        beamer_line = "\\usepackage[size=custom,height=%d,width=%d,scale=%.1f]{beamerposter}\n"
        header_2 = r"""% instead and customize the height and width (in cm) to fit.
% Large images may run out of memory quickly.
% To fix this use the LuaLaTeX compiler, which dynamically
% allocates memory.
\usepackage[braket, qm]{qcircuit}
\usepackage{amsmath}
\pdfmapfile{+sansmathaccent.map}
% \usepackage[landscape]{geometry}
% Comment out the above line if using the beamer documentclass.
\begin{document}
\begin{equation*}"""
        qcircuit_line = r"""
    \Qcircuit @C=%.1fem @R=%.1fem @!R {
"""
        output = StringIO()
        output.write(header_1)
        output.write('%% img_width = %d, img_depth = %d\n' % (self.img_width, self.img_depth))
        output.write(beamer_line % self._get_beamer_page())
        output.write(header_2)
        output.write(qcircuit_line %
                     (self.column_separation, self.row_separation))
        for i in range(self.img_width):
            output.write("\t \t")
            for j in range(self.img_depth + 1):
                cell_str = self._latex[i][j]
                # Don't truncate offset float if drawing a barrier
                if 'barrier' in cell_str:
                    output.write(cell_str)
                else:
                    # floats can cause "Dimension too large" latex error in
                    # xymatrix this truncates floats to avoid issue.
                    cell_str = re.sub(r'[-+]?\d*\.\d{2,}|\d{2,}',
                                      _truncate_float,
                                      cell_str)
                    output.write(cell_str)
                if j != self.img_depth:
                    output.write(" & ")
                else:
                    output.write(r'\\'+'\n')
        output.write('\t }\n')
        output.write('\\end{equation*}\n\n')
        output.write('\\end{document}')
        contents = output.getvalue()
        output.close()
        return contents

    def _initialize_latex_array(self, aliases=None):
        # pylint: disable=unused-argument
        self.img_depth, self.sum_column_widths = self._get_image_depth(aliases)
        self.sum_row_heights = self.img_width
        # choose the most compact row spacing, while not squashing them
        if self.has_box:
            self.row_separation = 0.0
        elif self.has_target:
            self.row_separation = 0.2
        else:
            self.row_separation = 1.0
        self._latex = [
            ["\\cw" if self.wire_type[self.ordered_regs[j]]
             else "\\qw" for i in range(self.img_depth + 1)]
            for j in range(self.img_width)]
        self._latex.append([" "] * (self.img_depth + 1))
        for i in range(self.img_width):
            if self.wire_type[self.ordered_regs[i]]:
                self._latex[i][0] = "\\lstick{" + self.ordered_regs[i][0] + \
                                    "_{" + str(self.ordered_regs[i][1]) + "}" + \
                                    ": 0}"
            else:
                self._latex[i][0] = "\\lstick{" + \
                                    self.ordered_regs[i][0] + "_{" + \
                                    str(self.ordered_regs[i][1]) + "}" + \
                                    ": \\ket{0}}"

    def _get_image_depth(self, aliases=None):
        """Get depth information for the circuit.

        Args:
            aliases (dict): dict mapping the current qubits in the circuit to
                new qubit names.

        Returns:
            int: number of columns in the circuit
            int: total size of columns in the circuit

        Raises:
            QISKitError: if trying to draw unsupported gates
        """
        columns = 2     # wires in the beginning and end
        is_occupied = [False] * self.img_width
        max_column_width = {}
        for op in self.circuit['instructions']:
            # useful information for determining row spacing
            boxed_gates = ['u0', 'u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's', 'sdg',
                           't', 'tdg', 'rx', 'ry', 'rz', 'ch', 'cy', 'crz', 'cu3']
            target_gates = ['cx', 'ccx']
            if op['name'] in boxed_gates:
                self.has_box = True
            if op['name'] in target_gates:
                self.has_target = True

            # useful information for determining column widths and final image scaling
            if op['name'] not in ['measure', 'reset', 'barrier']:
                qarglist = [self.qubit_list[i] for i in op['qubits']]
                if aliases is not None:
                    qarglist = map(lambda x: aliases[x], qarglist)
                if len(qarglist) == 1:
                    pos_1 = self.img_regs[(qarglist[0][0],
                                           qarglist[0][1])]
                    if 'conditional' in op:
                        mask = int(op['conditional']['mask'], 16)
                        if self._style.reverse:
                            mask = self._convert_mask(mask)
                        cl_reg = self.clbit_list[self._ffs(mask)]
                        if_reg = cl_reg[0]
                        pos_2 = self.img_regs[cl_reg]
                        for i in range(pos_1, pos_2 + self.cregs[if_reg]):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(pos_1, pos_2 + 1):
                                    is_occupied[j] = True
                                break
                    else:
                        if is_occupied[pos_1] is False:
                            is_occupied[pos_1] = True
                        else:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[pos_1] = True
                elif len(qarglist) == 2:
                    pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                    pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]

                    if 'conditional' in op:
                        mask = int(op['conditional']['mask'], 16)
                        if self._style.reverse:
                            mask = self._convert_mask(mask)
                        cl_reg = self.clbit_list[self._ffs(mask)]
                        if_reg = cl_reg[0]
                        pos_3 = self.img_regs[(if_reg, 0)]
                        if pos_1 > pos_2:
                            for i in range(pos_2, pos_3 + self.cregs[if_reg]):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(pos_2, pos_3 + 1):
                                        is_occupied[j] = True
                                    break
                        else:
                            for i in range(pos_1, pos_3 + self.cregs[if_reg]):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(pos_1, pos_3 + 1):
                                        is_occupied[j] = True
                                    break
                        # symetric gates have angle labels
                        if op['name'] in ['cu1']:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[max(pos_1, pos_2)] = True
                    else:
                        temp = [pos_1, pos_2]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[1]

                        for i in range(top, bottom + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, bottom + 1):
                                    is_occupied[j] = True
                                break
                        # symetric gates have angle labels
                        if op['name'] in ['cu1']:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[top] = True

                elif len(qarglist) == 3:
                    pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                    pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]
                    pos_3 = self.img_regs[(qarglist[2][0], qarglist[2][1])]

                    if 'conditional' in op:
                        mask = int(op['conditional']['mask'], 16)
                        if self._style.reverse:
                            mask = self._convert_mask(mask)
                        cl_reg = self.clbit_list[self._ffs(mask)]
                        if_reg = cl_reg[0]
                        pos_4 = self.img_regs[(if_reg, 0)]

                        temp = [pos_1, pos_2, pos_3, pos_4]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[2]

                        for i in range(top, pos_4 + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, pos_4 + 1):
                                    is_occupied[j] = True
                                break
                    else:
                        temp = [pos_1, pos_2, pos_3]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[2]

                        for i in range(top, bottom + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, bottom + 1):
                                    is_occupied[j] = True
                                break

                # update current column width
                arg_str_len = 0
                for arg in op['texparams']:
                    arg_str = re.sub(r'[-+]?\d*\.\d{2,}|\d{2,}', _truncate_float, arg)
                    arg_str_len += len(arg_str)
                if columns not in max_column_width:
                    max_column_width[columns] = 0
                max_column_width[columns] = max(arg_str_len,
                                                max_column_width[columns])
            elif op['name'] == "measure":
                assert len(op['clbits']) == 1 and len(op['qubits']) == 1
                if 'conditional' in op:
                    raise QISKitError('conditional measures currently not supported.')
                qname, qindex = self.total_2_register_index(
                    op['qubits'][0], self.qregs)
                cname, cindex = self.total_2_register_index(
                    op['clbits'][0], self.cregs)
                if aliases:
                    newq = aliases[(qname, qindex)]
                    qname = newq[0]
                    qindex = newq[1]
                pos_1 = self.img_regs[(qname, qindex)]
                pos_2 = self.img_regs[(cname, cindex)]
                temp = [pos_1, pos_2]
                temp.sort(key=int)
                [pos_1, pos_2] = temp
                for i in range(pos_1, pos_2 + 1):
                    if is_occupied[i] is False:
                        is_occupied[i] = True
                    else:
                        columns += 1
                        is_occupied = [False] * self.img_width
                        for j in range(pos_1, pos_2 + 1):
                            is_occupied[j] = True
                        break
                # update current column width
                if columns not in max_column_width:
                    max_column_width[columns] = 0
            elif op['name'] == "reset":
                if 'conditional' in op:
                    raise QISKitError('conditional reset currently not supported.')
                qname, qindex = self.total_2_register_index(
                    op['qubits'][0], self.qregs)
                if aliases:
                    newq = aliases[(qname, qindex)]
                    qname = newq[0]
                    qindex = newq[1]
                pos_1 = self.img_regs[(qname, qindex)]
                if is_occupied[pos_1] is False:
                    is_occupied[pos_1] = True
                else:
                    columns += 1
                    is_occupied = [False] * self.img_width
                    is_occupied[pos_1] = True
            elif op['name'] == "barrier":
                pass
            else:
                assert False, "bad node data"
        # every 3 characters is roughly one extra 'unit' of width in the cell
        # the gate name is 1 extra 'unit'
        # the qubit/cbit labels plus initial states is 2 more
        # the wires poking out at the ends is 2 more
        sum_column_widths = sum(1 + v / 3 for v in max_column_width.values())
        return columns+1, ceil(sum_column_widths)+4

    def _get_beamer_page(self):
        """Get height, width & scale attributes for the beamer page.

        Returns:
            tuple: (height, width, scale) desirable page attributes
        """
        # PIL python package limits image size to around a quarter gigabyte
        # this means the beamer image should be limited to < 50000
        # if you want to avoid a "warning" too, set it to < 25000
        PIL_limit = 40000

        # the beamer latex template limits each dimension to < 19 feet (i.e. 575cm)
        beamer_limit = 550

        # columns are roughly twice as big as rows
        aspect_ratio = self.sum_row_heights / self.sum_column_widths

        # choose a page margin so circuit is not cropped
        margin_factor = 1.5
        height = min(self.sum_row_heights * margin_factor, beamer_limit)
        width = min(self.sum_column_widths * margin_factor, beamer_limit)

        # if too large, make it fit
        if height * width > PIL_limit:
            height = min(np.sqrt(PIL_limit * aspect_ratio), beamer_limit)
            width = min(np.sqrt(PIL_limit / aspect_ratio), beamer_limit)

        # if too small, give it a minimum size
        height = max(height, 10)
        width = max(width, 10)

        return (height, width, self.scale)

    def total_2_register_index(self, index, registers):
        """Get register name for qubit index.

        This function uses the self.qregs ordered dictionary, which looks like
        {'qr1': 2, 'qr2', 3}
        to get the register name for the total qubit index. For the above example,
        index in [0,1] returns 'qr1' and index in [2,4] returns 'qr2'.

        Args:
            index (int): total qubit index among all quantum registers
            registers (OrderedDict): OrderedDict as described above.
        Returns:
            str: name of register associated with qubit index.
        Raises:
            ValueError: if the qubit index lies outside the range of qubit
                registers.
        """
        count = 0
        for name, size in registers.items():
            if count + size > index:
                return name, index - count
            else:
                count += size
        raise ValueError('qubit index lies outside range of qubit registers')

    def _convert_mask(self, mask):
        orig_clbit_list = []
        for cr in self.orig_cregs:
            for i in range(self.orig_cregs[cr]):
                orig_clbit_list.append((cr, i))
        bit_list = [(mask >> bit) & 1 for bit in range(
            len(orig_clbit_list) - 1, -1, -1)]
        converted_mask_list = [None] * len(bit_list)
        converted_mask = 0
        for pos, bit in enumerate(reversed(bit_list)):
            new_pos = self.clbit_list.index(orig_clbit_list[pos])
            converted_mask_list[new_pos] = bit
        if None in converted_mask_list:
            raise VisualizationError('Reverse mask creation failed')
        converted_mask_list = list(reversed(converted_mask_list))
        for bit in converted_mask_list:
            converted_mask = (converted_mask << 1) | bit
        return converted_mask

    def _build_latex_array(self, aliases=None):
        """Returns an array of strings containing \\LaTeX for this circuit.

        If aliases is not None, aliases contains a dict mapping
        the current qubits in the circuit to new qubit names.
        We will deduce the register names and sizes from aliases.
        """
        columns = 1
        is_occupied = [False] * self.img_width

        # Rename qregs if necessary
        if aliases:
            qregdata = {}
            for q in aliases.values():
                if q[0] not in qregdata:
                    qregdata[q[0]] = q[1] + 1
                elif qregdata[q[0]] < q[1] + 1:
                    qregdata[q[0]] = q[1] + 1
        else:
            qregdata = self.qregs

        for _, op in enumerate(self.circuit['instructions']):
            if 'conditional' in op:
                mask = int(op['conditional']['mask'], 16)
                if self._style.reverse:
                    mask = self._convert_mask(mask)
                cl_reg = self.clbit_list[self._ffs(mask)]
                if_reg = cl_reg[0]
                pos_2 = self.img_regs[cl_reg]
                if_value = format(int(op['conditional']['val'], 16),
                                  'b').zfill(self.cregs[if_reg])[::-1]
            if op['name'] not in ['measure', 'barrier']:
                nm = op['name']
                qarglist = [self.qubit_list[i] for i in op['qubits']]
                if aliases is not None:
                    qarglist = map(lambda x: aliases[x], qarglist)
                if len(qarglist) == 1:
                    pos_1 = self.img_regs[(qarglist[0][0],
                                           qarglist[0][1])]
                    if 'conditional' in op:
                        mask = int(op['conditional']['mask'], 16)
                        if self._style.reverse:
                            mask = self._convert_mask(mask)
                        cl_reg = self.clbit_list[self._ffs(mask)]
                        if_reg = cl_reg[0]
                        pos_2 = self.img_regs[cl_reg]
                        for i in range(pos_1, pos_2 + self.cregs[if_reg]):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(pos_1, pos_2 + 1):
                                    is_occupied[j] = True
                                break

                        if nm == "x":
                            self._latex[pos_1][columns] = "\\gate{X}"
                        elif nm == "y":
                            self._latex[pos_1][columns] = "\\gate{Y}"
                        elif nm == "z":
                            self._latex[pos_1][columns] = "\\gate{Z}"
                        elif nm == "h":
                            self._latex[pos_1][columns] = "\\gate{H}"
                        elif nm == "s":
                            self._latex[pos_1][columns] = "\\gate{S}"
                        elif nm == "sdg":
                            self._latex[pos_1][columns] = "\\gate{S^\\dag}"
                        elif nm == "t":
                            self._latex[pos_1][columns] = "\\gate{T}"
                        elif nm == "tdg":
                            self._latex[pos_1][columns] = "\\gate{T^\\dag}"
                        elif nm == "u0":
                            self._latex[pos_1][columns] = "\\gate{U_0(%s)}" % (
                                op["texparams"][0])
                        elif nm == "u1":
                            self._latex[pos_1][columns] = "\\gate{U_1(%s)}" % (
                                op["texparams"][0])
                        elif nm == "u2":
                            self._latex[pos_1][columns] =\
                                "\\gate{U_2\\left(%s,%s\\right)}" % (
                                    op["texparams"][0], op["texparams"][1])
                        elif nm == "u3":
                            self._latex[pos_1][columns] = "\\gate{U_3(%s,%s,%s)}" \
                                % (op["texparams"][0], op["texparams"][1], op["texparams"][2])
                        elif nm == "rx":
                            self._latex[pos_1][columns] = "\\gate{R_x(%s)}" % (
                                op["texparams"][0])
                        elif nm == "ry":
                            self._latex[pos_1][columns] = "\\gate{R_y(%s)}" % (
                                op["texparams"][0])
                        elif nm == "rz":
                            self._latex[pos_1][columns] = "\\gate{R_z(%s)}" % (
                                op["texparams"][0])

                        gap = pos_2 - pos_1
                        for i in range(self.cregs[if_reg]):
                            if if_value[i] == '1':
                                self._latex[pos_2 + i][columns] = \
                                    "\\control \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1
                            else:
                                self._latex[pos_2 + i][columns] = \
                                    "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1

                    else:
                        if not is_occupied[pos_1]:
                            is_occupied[pos_1] = True
                        else:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[pos_1] = True

                        if nm == "x":
                            self._latex[pos_1][columns] = "\\gate{X}"
                        elif nm == "y":
                            self._latex[pos_1][columns] = "\\gate{Y}"
                        elif nm == "z":
                            self._latex[pos_1][columns] = "\\gate{Z}"
                        elif nm == "h":
                            self._latex[pos_1][columns] = "\\gate{H}"
                        elif nm == "s":
                            self._latex[pos_1][columns] = "\\gate{S}"
                        elif nm == "sdg":
                            self._latex[pos_1][columns] = "\\gate{S^\\dag}"
                        elif nm == "t":
                            self._latex[pos_1][columns] = "\\gate{T}"
                        elif nm == "tdg":
                            self._latex[pos_1][columns] = "\\gate{T^\\dag}"
                        elif nm == "u0":
                            self._latex[pos_1][columns] = "\\gate{U_0(%s)}" % (
                                op["texparams"][0])
                        elif nm == "u1":
                            self._latex[pos_1][columns] = "\\gate{U_1(%s)}" % (
                                op["texparams"][0])
                        elif nm == "u2":
                            self._latex[pos_1][columns] = \
                                "\\gate{U_2\\left(%s,%s\\right)}" % (
                                    op["texparams"][0], op["texparams"][1])
                        elif nm == "u3":
                            self._latex[pos_1][columns] = "\\gate{U_3(%s,%s,%s)}" \
                                % (op["texparams"][0], op["texparams"][1], op["texparams"][2])
                        elif nm == "rx":
                            self._latex[pos_1][columns] = "\\gate{R_x(%s)}" % (
                                op["texparams"][0])
                        elif nm == "ry":
                            self._latex[pos_1][columns] = "\\gate{R_y(%s)}" % (
                                op["texparams"][0])
                        elif nm == "rz":
                            self._latex[pos_1][columns] = "\\gate{R_z(%s)}" % (
                                op["texparams"][0])
                        elif nm == "reset":
                            self._latex[pos_1][columns] = \
                                "\\push{\\rule{.6em}{0em}\\ket{0}\\rule{.2em}{0em}} \\qw"

                elif len(qarglist) == 2:
                    pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                    pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]

                    if 'conditional' in op:
                        pos_3 = self.img_regs[(if_reg, 0)]
                        temp = [pos_1, pos_2, pos_3]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[1]

                        for i in range(top, pos_3 + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, pos_3 + 1):
                                    is_occupied[j] = True
                                break
                        # symetric gates have angle labels
                        if op['name'] in ['cu1']:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[top] = True

                        gap = pos_3 - bottom
                        for i in range(self.cregs[if_reg]):
                            if if_value[i] == '1':
                                self._latex[pos_3 + i][columns] = \
                                    "\\control \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1
                            else:
                                self._latex[pos_3 + i][columns] = \
                                    "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1

                        if nm == "cx":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\targ"
                        elif nm == "cz":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\control\\qw"
                        elif nm == "cy":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{Y}"
                        elif nm == "ch":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{H}"
                        elif nm == "swap":
                            self._latex[pos_1][columns] = "\\qswap"
                            self._latex[pos_2][columns] = \
                                "\\qswap \\qwx[" + str(pos_1 - pos_2) + "]"
                        elif nm == "crz":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = \
                                "\\gate{R_z(%s)}" % (op["texparams"][0])
                        elif nm == "cu1":
                            self._latex[pos_1][columns-1] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns-1] = "\\control\\qw"
                            self._latex[min(pos_1, pos_2)][columns] = \
                                "\\dstick{%s}\\qw" % (op["texparams"][0])
                            self._latex[max(pos_1, pos_2)][columns] = "\\qw"
                        elif nm == "cu3":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = \
                                "\\gate{U_3(%s,%s,%s)}" % (op["texparams"][0],
                                                           op["texparams"][1],
                                                           op["texparams"][2])
                    else:
                        temp = [pos_1, pos_2]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[1]

                        for i in range(top, bottom + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, bottom + 1):
                                    is_occupied[j] = True
                                break
                        # symetric gates have angle labels
                        if op['name'] in ['cu1']:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[top] = True

                        if nm == "cx":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\targ"
                        elif nm == "cz":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\control\\qw"
                        elif nm == "cy":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{Y}"
                        elif nm == "ch":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{H}"
                        elif nm == "swap":
                            self._latex[pos_1][columns] = "\\qswap"
                            self._latex[pos_2][columns] = \
                                "\\qswap \\qwx[" + str(pos_1 - pos_2) + "]"
                        elif nm == "crz":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = \
                                "\\gate{R_z(%s)}" % (op["texparams"][0])
                        elif nm == "cu1":
                            self._latex[pos_1][columns-1] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns-1] = "\\control\\qw"
                            self._latex[min(pos_1, pos_2)][columns] = \
                                "\\dstick{%s}\\qw" % (op["texparams"][0])
                            self._latex[max(pos_1, pos_2)][columns] = "\\qw"
                        elif nm == "cu3":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{U_3(%s,%s,%s)}" \
                                % (op["texparams"][0], op["texparams"][1], op["texparams"][2])

                elif len(qarglist) == 3:
                    pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                    pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]
                    pos_3 = self.img_regs[(qarglist[2][0], qarglist[2][1])]

                    if 'conditional' in op:
                        pos_4 = self.img_regs[(if_reg, 0)]

                        temp = [pos_1, pos_2, pos_3, pos_4]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[2]

                        for i in range(top, pos_4 + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, pos_4 + 1):
                                    is_occupied[j] = True
                                break

                        gap = pos_4 - bottom
                        for i in range(self.cregs[if_reg]):
                            if if_value[i] == '1':
                                self._latex[pos_4 + i][columns] = \
                                    "\\control \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1
                            else:
                                self._latex[pos_4 + i][columns] = \
                                    "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1

                        if nm == "ccx":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\ctrl{" + str(pos_3 - pos_2) + "}"
                            self._latex[pos_3][columns] = "\\targ"

                        if nm == "cswap":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\qswap"
                            self._latex[pos_3][columns] = \
                                "\\qswap \\qwx[" + str(pos_2 - pos_3) + "]"
                    else:
                        temp = [pos_1, pos_2, pos_3]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[2]

                        for i in range(top, bottom + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, bottom + 1):
                                    is_occupied[j] = True
                                break

                        if nm == "ccx":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\ctrl{" + str(pos_3 - pos_2) + "}"
                            self._latex[pos_3][columns] = "\\targ"

                        if nm == "cswap":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\qswap"
                            self._latex[pos_3][columns] = \
                                "\\qswap \\qwx[" + str(pos_2 - pos_3) + "]"

            elif op["name"] == "measure":
                assert len(op['clbits']) == 1 and \
                    len(op['qubits']) == 1 and \
                    'params' not in op, "bad operation record"

                if 'conditional' in op:
                    assert False, "If controlled measures currently not supported."
                qname, qindex = self.total_2_register_index(
                    op['qubits'][0], self.qregs)
                cname, cindex = self.total_2_register_index(
                    op['clbits'][0], self.cregs)

                if aliases:
                    newq = aliases[(qname, qindex)]
                    qname = newq[0]
                    qindex = newq[1]

                pos_1 = self.img_regs[(qname, qindex)]
                pos_2 = self.img_regs[(cname, cindex)]

                for i in range(pos_1, pos_2 + 1):
                    if is_occupied[i] is False:
                        is_occupied[i] = True
                    else:
                        columns += 1
                        is_occupied = [False] * self.img_width
                        for j in range(pos_1, pos_2 + 1):
                            is_occupied[j] = True
                        break

                try:
                    self._latex[pos_1][columns] = "\\meter"
                    prev_column = [x[columns - 1] for x in self._latex]
                    for item, prev_entry in enumerate(prev_column):
                        if 'barrier' in prev_entry:
                            span = re.search('barrier{(.*)}', prev_entry)
                            if span and (
                                    item + int(span.group(1))) - pos_1 >= 0:
                                self._latex[
                                    item][columns - 1] = prev_entry.replace(
                                        '\\barrier{', '\\barrier[-1.15em]{')

                    self._latex[pos_2][columns] = \
                        "\\cw \\cwx[-" + str(pos_2 - pos_1) + "]"
                except Exception as e:
                    raise QISKitError('Error during Latex building: %s' %
                                      str(e))
            elif op['name'] == "barrier":
                qarglist = [self.qubit_list[i] for i in op['qubits']]
                if aliases is not None:
                    qarglist = map(lambda x: aliases[x], qarglist)
                start = self.img_regs[(qarglist[0][0],
                                       qarglist[0][1])]
                span = len(op['qubits']) - 1
                self._latex[start][columns] += " \\barrier{" + str(span) + "}"
            else:
                assert False, "bad node data"

    def _ffs(self, mask):
        """Find index of first set bit.

        Args:
            mask (int): integer to search
        Returns:
            int: index of the first set bit.
        """
        origin = (mask & (-mask)).bit_length()
        if self._style.reverse:
            return origin + 1
        return origin - 1


def _get_register_specs(bit_labels):
    """
    Get the number and size of unique registers from bit_labels list.

    Args:
        bit_labels (list): this list is of the form::

            [['reg1', 0], ['reg1', 1], ['reg2', 0]]

            which indicates a register named "reg1" of size 2
            and a register named "reg2" of size 1. This is the
            format of classic and quantum bit labels in qobj
            header.

    Yields:
        tuple: iterator of register_name:size pairs.
    """
    it = groupby(bit_labels, operator.itemgetter(0))
    for register_name, sub_it in it:
        yield register_name, max(ind[1] for ind in sub_it) + 1


def _truncate_float(matchobj, format_str='0.2g'):
    """Truncate long floats

    Args:
        matchobj (re.Match): contains original float
        format_str (str): format specifier
    Returns:
       str: returns truncated float
    """
    if matchobj.group(0):
        return format(float(matchobj.group(0)), format_str)
    return ''


# -----------------------------------------------------------------------------
# matplotlib_circuit_drawer
# -----------------------------------------------------------------------------
WID = 0.65
HIG = 0.65
DEFAULT_SCALE = 4.3
PORDER_GATE = 5
PORDER_LINE = 2
PORDER_GRAY = 3
PORDER_TEXT = 6
PORDER_SUBP = 4


def matplotlib_circuit_drawer(circuit,
                              basis='id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,'
                                    'cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap',
                              scale=0.7,
                              filename=None,
                              style=None):
    """Draw a quantum circuit based on matplotlib.
    If `%matplotlib inline` is invoked in a Jupyter notebook, it visualizes a circuit inline.
    We recommend `%config InlineBackend.figure_format = 'svg'` for the inline visualization.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        basis (str): comma separated list of gates
        scale (float): scaling factor
        filename (str): file path to save image to
        style (dict or str): dictionary of style or file name of style file

    Returns:
        PIL.Image: an in-memory representation of the circuit diagram
    """
    if ',' not in basis:
        logger.warning('Warning: basis is not comma separated: "%s". '
                       'Perhaps you set `filename` to `basis`.', basis)
    qcd = MatplotlibDrawer(basis=basis, scale=scale, style=style)
    qcd.parse_circuit(circuit)
    return qcd.draw(filename)


Register = namedtuple('Register', 'name index')


class Anchor:
    def __init__(self, reg_num, yind, fold):
        self.__yind = yind
        self.__fold = fold
        self.__reg_num = reg_num
        self.__gate_placed = []

    def plot_coord(self, index, gate_width):
        h_pos = index % self.__fold + 1
        # check folding
        if self.__fold > 0:
            if h_pos + (gate_width - 1) > self.__fold:
                index += self.__fold - (h_pos - 1)
            x_pos = index % self.__fold + 1 + 0.5 * (gate_width - 1)
            y_pos = self.__yind - (index // self.__fold) * (self.__reg_num + 1)
        else:
            x_pos = index + 1 + 0.5 * (gate_width - 1)
            y_pos = self.__yind

        return x_pos, y_pos

    def is_locatable(self, index, gate_width):
        hold = [index + i for i in range(gate_width)]
        for p in hold:
            if p in self.__gate_placed:
                return False
        return True

    def set_index(self, index, gate_width):
        h_pos = index % self.__fold + 1
        if h_pos + (gate_width - 1) > self.__fold:
            _index = index + self.__fold - (h_pos - 1)
        else:
            _index = index
        for ii in range(gate_width):
            if _index + ii not in self.__gate_placed:
                self.__gate_placed.append(_index + ii)
        self.__gate_placed.sort()

    def get_index(self):
        if self.__gate_placed:
            return self.__gate_placed[-1] + 1
        return 0


class MatplotlibDrawer:
    def __init__(self,
                 basis='id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,'
                       'cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap',
                 scale=1.0, style=None):

        self._ast = None
        self._basis = basis
        self._scale = DEFAULT_SCALE * scale
        self._creg = []
        self._qreg = []
        self._ops = []
        self._qreg_dict = OrderedDict()
        self._creg_dict = OrderedDict()
        self._cond = {
            'n_lines': 0,
            'xmax': 0,
            'ymax': 0,
        }

        self._style = QCStyle()
        if style:
            if isinstance(style, dict):
                self._style.set_style(style)
            elif isinstance(style, str):
                with open(style, 'r') as infile:
                    dic = json.load(infile)
                self._style.set_style(dic)

        self.figure = plt.figure()
        self.figure.patch.set_facecolor(color=self._style.bg)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    def load_qasm_file(self, filename):
        circuit = load_qasm_file(filename, name='draw', basis_gates=self._basis)
        self.parse_circuit(circuit)

    def parse_circuit(self, circuit):
        dag_circuit = DAGCircuit.fromQuantumCircuit(circuit, expand_gates=False)
        self._ast = transpile(dag_circuit, basis_gates=self._basis, format='json')
        self._registers()
        self._ops = self._ast['instructions']

    def _registers(self):
        # NOTE: formats of clbit and qubit are different!
        header = self._ast['header']
        self._creg = []
        for e in header['clbit_labels']:
            for i in range(e[1]):
                self._creg.append(Register(name=e[0], index=i))
        assert len(self._creg) == header['number_of_clbits']
        self._qreg = []
        for e in header['qubit_labels']:
            self._qreg.append(Register(name=e[0], index=e[1]))
        assert len(self._qreg) == header['number_of_qubits']

    @property
    def ast(self):
        return self._ast

    def _gate(self, xy, fc=None, wide=False, text=None, subtext=None):
        xpos, ypos = xy

        if wide:
            wid = WID * 2.8
        else:
            wid = WID
        if fc:
            _fc = fc
        elif text:
            _fc = self._style.dispcol[text]
        else:
            _fc = self._style.gc

        box = patches.Rectangle(xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG), width=wid, height=HIG,
                                fc=_fc, ec=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        if text:
            disp_text = "${}$".format(self._style.disptex[text])
            if subtext:
                self.ax.text(xpos, ypos + 0.15 * HIG, disp_text, ha='center', va='center',
                             fontsize=self._style.fs,
                             color=self._style.gt,
                             clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos - 0.3 * HIG, subtext, ha='center', va='center',
                             fontsize=self._style.sfs,
                             color=self._style.sc,
                             clip_on=True,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos, disp_text, ha='center', va='center',
                             fontsize=self._style.fs,
                             color=self._style.gt,
                             clip_on=True,
                             zorder=PORDER_TEXT)

    def _subtext(self, xy, text):
        xpos, ypos = xy

        self.ax.text(xpos, ypos - 0.3 * HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs,
                     color=self._style.tc,
                     clip_on=True,
                     zorder=PORDER_TEXT)

    def _line(self, xy0, xy1, lc=None, ls=None):
        x0, y0 = xy0
        x1, y1 = xy1
        if lc is None:
            linecolor = self._style.lc
        else:
            linecolor = lc
        if ls is None:
            linestyle = 'solid'
        else:
            linestyle = ls
        if linestyle == 'doublet':
            theta = np.arctan2(np.abs(x1 - x0), np.abs(y1 - y0))
            dx = 0.05 * WID * np.cos(theta)
            dy = 0.05 * WID * np.sin(theta)
            self.ax.plot([x0 + dx, x1 + dx], [y0 + dy, y1 + dy],
                         color=linecolor,
                         linewidth=1.0,
                         linestyle='solid',
                         zorder=PORDER_LINE)
            self.ax.plot([x0 - dx, x1 - dx], [y0 - dy, y1 - dy],
                         color=linecolor,
                         linewidth=1.0,
                         linestyle='solid',
                         zorder=PORDER_LINE)
        else:
            self.ax.plot([x0, x1], [y0, y1],
                         color=linecolor,
                         linewidth=1.0,
                         linestyle=linestyle,
                         zorder=PORDER_LINE)

    def _measure(self, qxy, cxy, cid):
        qx, qy = qxy
        cx, cy = cxy

        self._gate(qxy, fc=self._style.dispcol['meas'])
        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7, height=HIG * 0.7,
                          theta1=0, theta2=180, fill=False,
                          ec=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([qx, qx + 0.35 * WID], [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        # arrow
        self._line(qxy, [cx, cy+0.35*WID], lc=self._style.cc, ls=self._style.cline)
        arrowhead = patches.Polygon(((cx-0.20*WID, cy+0.35*WID),
                                     (cx+0.20*WID, cy+0.35*WID),
                                     (cx, cy)),
                                    fc=self._style.cc,
                                    ec=None)
        self.ax.add_artist(arrowhead)
        # target
        if self._style.bundle:
            self.ax.text(cx + .25, cy + .1, str(cid), ha='left', va='bottom',
                         fontsize=0.8 * self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)

    def _conds(self, xy, istrue=False):
        xpos, ypos = xy

        if istrue:
            _fc = self._style.lc
        else:
            _fc = self._style.gc

        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=_fc, ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _ctrl_qubit(self, xy):
        xpos, ypos = xy

        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=self._style.lc, ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _tgt_qubit(self, xy):
        xpos, ypos = xy

        box = patches.Circle(xy=(xpos, ypos), radius=HIG * 0.35,
                             fc=self._style.dispcol['target'], ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)
        # add '+' symbol
        self.ax.plot([xpos, xpos], [ypos - 0.35 * HIG, ypos + 0.35 * HIG],
                     color=self._style.lc, linewidth=1.0, zorder=PORDER_GATE)
        self.ax.plot([xpos - 0.35 * HIG, xpos + 0.35 * HIG], [ypos, ypos],
                     color=self._style.lc, linewidth=1.0, zorder=PORDER_GATE)

    def _swap(self, xy):
        xpos, ypos = xy

        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID], [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_LINE)
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID], [ypos + 0.20 * WID, ypos - 0.20 * WID],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_LINE)

    def _barrier(self, config, anc):
        xys = config['coord']
        group = config['group']
        y_reg = []
        for qreg in self._qreg_dict.values():
            if qreg['group'] in group:
                y_reg.append(qreg['y'])
        x0 = xys[0][0]

        box_y0 = min(y_reg) - int(anc / self._style.fold) * (self._cond['n_lines'] + 1) - 0.5
        box_y1 = max(y_reg) - int(anc / self._style.fold) * (self._cond['n_lines'] + 1) + 0.5
        box = patches.Rectangle(xy=(x0 - 0.3 * WID, box_y0),
                                width=0.6 * WID, height=box_y1 - box_y0,
                                fc=self._style.bc, ec=None, alpha=0.6,
                                linewidth=1.5, zorder=PORDER_GRAY)
        self.ax.add_patch(box)
        for xy in xys:
            xpos, ypos = xy
            self.ax.plot([xpos, xpos], [ypos + 0.5, ypos - 0.5],
                         linewidth=1, linestyle="dashed",
                         color=self._style.lc,
                         zorder=PORDER_TEXT)

    def _linefeed_mark(self, xy):
        xpos, ypos = xy

        self.ax.plot([xpos - .1, xpos - .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, zorder=PORDER_LINE)
        self.ax.plot([xpos + .1, xpos + .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, zorder=PORDER_LINE)

    def draw(self, filename=None, verbose=False):
        self._draw_regs()
        self._draw_ops(verbose)
        _xl = - self._style.margin[0]
        _xr = self._cond['xmax'] + self._style.margin[1]
        _yb = - self._cond['ymax'] - self._style.margin[2] + 1 - 0.5
        _yt = self._style.margin[3] + 0.5
        self.ax.set_xlim(_xl, _xr)
        self.ax.set_ylim(_yb, _yt)
        # update figure size
        fig_w = _xr - _xl
        fig_h = _yt - _yb
        if self._style.figwidth < 0.0:
            self._style.figwidth = fig_w * self._scale * self._style.fs / 72 / WID
        self.figure.set_size_inches(self._style.figwidth, self._style.figwidth * fig_h / fig_w)

        if get_matplotlib_backend() == 'module://ipykernel.pylab.backend_inline':
            # returns None when matplotlib is inline mode to prevent Jupyter
            # with matplotlib inlining enabled to draw the diagram twice.
            im = None
        else:
            # when matplotlib is not inline mode,
            # self.figure.savefig is called twice because...
            # ... this is needed to get the in-memory representation
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, 'circuit.png')
                self.figure.savefig(tmpfile, dpi=self._style.dpi,
                                    bbox_inches='tight')
                im = Image.open(tmpfile)
                _trim(im)
                os.remove(tmpfile)

        # ... and this is needed to delegate in matplotlib the generation of
        # the proper format.
        if filename:
            self.figure.savefig(filename, dpi=self._style.dpi,
                                bbox_inches='tight')
        return im

    def _draw_regs(self):
        # quantum register
        for ii, reg in enumerate(self._qreg):
            if len(self._qreg) > 1:
                label = '${}_{{{}}}$'.format(reg.name, reg.index)
            else:
                label = '${}$'.format(reg.name)
            pos = -ii
            self._qreg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index, 'group': reg.name}
            self._cond['n_lines'] += 1
        # classical register
        if self._creg:
            n_creg = self._creg.copy()
            n_creg.pop(0)
            idx = 0
            y_off = -len(self._qreg)
            for ii, (reg, nreg) in enumerate(zip_longest(self._creg, n_creg)):
                pos = y_off - idx
                if self._style.bundle:
                    label = '${}$'.format(reg.name)
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index,
                                           'group': reg.name}
                    if not (not nreg or reg.name != nreg.name):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.name, reg.index)
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index,
                                           'group': reg.name}
                self._cond['n_lines'] += 1
                idx += 1
        # reverse bit order
        if self._style.reverse:
            self._reverse_bits(self._qreg_dict)
            self._reverse_bits(self._creg_dict)

    def _reverse_bits(self, target_dict):
        coord = {}
        # grouping
        for dict_ in target_dict.values():
            if dict_['group'] not in coord:
                coord[dict_['group']] = [dict_['y']]
            else:
                coord[dict_['group']].insert(0, dict_['y'])
        # reverse bit order
        for key in target_dict.keys():
            target_dict[key]['y'] = coord[target_dict[key]['group']].pop(0)

    def _draw_regs_sub(self, n_fold, feedline_l=False, feedline_r=False):
        # quantum register
        for qreg in self._qreg_dict.values():
            if n_fold == 0:
                label = qreg['label'] + ' : $\\left|0\\right\\rangle$'
            else:
                label = qreg['label']
            y = qreg['y'] - n_fold * (self._cond['n_lines'] + 1)
            self.ax.text(-0.5, y, label, ha='right', va='center',
                         fontsize=self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)
            self._line([0, y], [self._cond['xmax'], y])
        # classical register
        this_creg_dict = {}
        for creg in self._creg_dict.values():
            if n_fold == 0:
                label = creg['label'] + ' :  0 '
            else:
                label = creg['label']
            y = creg['y'] - n_fold * (self._cond['n_lines'] + 1)
            if y not in this_creg_dict.keys():
                this_creg_dict[y] = {'val': 1, 'label': label}
            else:
                this_creg_dict[y]['val'] += 1
        for y, this_creg in this_creg_dict.items():
            # bundle
            if this_creg['val'] > 1:
                self.ax.plot([.6, .7], [y - .1, y + .1],
                             color=self._style.cc,
                             zorder=PORDER_LINE)
                self.ax.text(0.5, y + .1, str(this_creg['val']), ha='left', va='bottom',
                             fontsize=0.8 * self._style.fs,
                             color=self._style.tc,
                             clip_on=True,
                             zorder=PORDER_TEXT)
            self.ax.text(-0.5, y, this_creg['label'], ha='right', va='center',
                         fontsize=self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)
            self._line([0, y], [self._cond['xmax'], y], lc=self._style.cc, ls=self._style.cline)

        # lf line
        if feedline_r:
            self._linefeed_mark((self._style.fold + 1 - 0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))
        if feedline_l:
            self._linefeed_mark((0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))

    def _draw_ops(self, verbose=False):
        _force_next = 'measure barrier'.split()
        _wide_gate = 'u2 u3 cu2 cu3'.split()
        _barriers = {'coord': [], 'group': []}
        next_ops = self._ops.copy()
        next_ops.pop(0)
        this_anc = 0

        #
        # generate coordinate manager
        #
        q_anchors = {}
        for key, qreg in self._qreg_dict.items():
            q_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=qreg['y'],
                                    fold=self._style.fold)
        c_anchors = {}
        for key, creg in self._creg_dict.items():
            c_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=creg['y'],
                                    fold=self._style.fold)
        #
        # draw gates
        #
        for i, (op, op_next) in enumerate(zip_longest(self._ops, next_ops)):
            # wide gate
            if op['name'] in _wide_gate:
                _iswide = True
                gw = 2
            else:
                _iswide = False
                gw = 1
            # get qreg index
            if 'qubits' in op.keys():
                q_idxs = op['qubits']
            else:
                q_idxs = []
            # get creg index
            if 'clbits' in op.keys():
                c_idxs = op['clbits']
            else:
                c_idxs = []
            # find empty space to place gate
            if not _barriers['group']:
                this_anc = max([q_anchors[ii].get_index() for ii in q_idxs])
                while True:
                    if op['name'] in _force_next or 'conditional' in op.keys() or \
                            not self._style.compress:
                        occupied = self._qreg_dict.keys()
                    else:
                        occupied = q_idxs
                    q_list = [ii for ii in range(min(occupied), max(occupied) + 1)]
                    locs = [q_anchors[jj].is_locatable(this_anc, gw) for jj in q_list]
                    if all(locs):
                        for ii in q_list:
                            if op['name'] == 'barrier' and not self._style.barrier:
                                q_anchors[ii].set_index(this_anc - 1, gw)
                            else:
                                q_anchors[ii].set_index(this_anc, gw)
                        break
                    else:
                        this_anc += 1
            # qreg coordinate
            q_xy = [q_anchors[ii].plot_coord(this_anc, gw) for ii in q_idxs]
            # creg corrdinate
            c_xy = [c_anchors[ii].plot_coord(this_anc, gw) for ii in c_idxs]
            # bottom and top point of qreg
            qreg_b = min(q_xy, key=lambda xy: xy[1])
            qreg_t = max(q_xy, key=lambda xy: xy[1])

            if verbose:
                print(i, op)

            # rotation parameter
            if 'params' in op.keys():
                param = self.param_parse(op['params'], self._style.pimode)
            else:
                param = None
            # conditional gate
            if 'conditional' in op.keys():
                c_xy = [c_anchors[ii].plot_coord(this_anc, gw) for ii in self._creg_dict]
                # cbit list to consider
                fmt_c = '{{:0{}b}}'.format(len(c_xy))
                mask = int(op['conditional']['mask'], 16)
                cmask = list(fmt_c.format(mask))[::-1]
                # value
                fmt_v = '{{:0{}b}}'.format(cmask.count('1'))
                val = int(op['conditional']['val'], 16)
                vlist = list(fmt_v.format(val))[::-1]
                # plot conditionals
                v_ind = 0
                xy_plot = []
                for xy, m in zip(c_xy, cmask):
                    if m == '1':
                        if xy not in xy_plot:
                            if vlist[v_ind] == '1' or self._style.bundle:
                                self._conds(xy, istrue=True)
                            else:
                                self._conds(xy, istrue=False)
                            xy_plot.append(xy)
                        v_ind += 1
                creg_b = sorted(xy_plot, key=lambda xy: xy[1])[0]
                self._subtext(creg_b, op['conditional']['val'])
                self._line(qreg_t, creg_b, lc=self._style.cc, ls=self._style.cline)
            #
            # draw special gates
            #
            if op['name'] == 'measure':
                vv = self._creg_dict[c_idxs[0]]['index']
                self._measure(q_xy[0], c_xy[0], vv)
            elif op['name'] == 'barrier':
                q_group = self._qreg_dict[q_idxs[0]]['group']
                if q_group not in _barriers['group']:
                    _barriers['group'].append(q_group)
                _barriers['coord'].append(q_xy[0])
                if op_next and op_next['name'] == 'barrier':
                    continue
                else:
                    if self._style.barrier:
                        self._barrier(_barriers, this_anc)
                    _barriers['group'].clear()
                    _barriers['coord'].clear()
            #
            # draw single qubit gates
            #
            elif len(q_xy) == 1:
                disp = op['name']
                if param:
                    self._gate(q_xy[0], wide=_iswide, text=disp, subtext='{}'.format(param))
                else:
                    self._gate(q_xy[0], wide=_iswide, text=disp)
            #
            # draw multi-qubit gates (n=2)
            #
            elif len(q_xy) == 2:
                # cx
                if op['name'] in ['cx']:
                    self._ctrl_qubit(q_xy[0])
                    self._tgt_qubit(q_xy[1])
                # cz for latexmode
                elif op['name'] == 'cz':
                    if self._style.latexmode:
                        self._ctrl_qubit(q_xy[0])
                        self._ctrl_qubit(q_xy[1])
                    else:
                        disp = op['name'].replace('c', '')
                        self._ctrl_qubit(q_xy[0])
                        self._gate(q_xy[1], wide=_iswide, text=disp)
                # control gate
                elif op['name'] in ['cy', 'ch', 'cu3', 'crz']:
                    disp = op['name'].replace('c', '')
                    self._ctrl_qubit(q_xy[0])
                    if param:
                        self._gate(q_xy[1], wide=_iswide, text=disp, subtext='{}'.format(param))
                    else:
                        self._gate(q_xy[1], wide=_iswide, text=disp)
                # cu1 for latexmode
                elif op['name'] in ['cu1']:
                    disp = op['name'].replace('c', '')
                    self._ctrl_qubit(q_xy[0])
                    if self._style.latexmode:
                        self._ctrl_qubit(q_xy[1])
                        self._subtext(qreg_b, param)
                    else:
                        self._gate(q_xy[1], wide=_iswide, text=disp, subtext='{}'.format(param))
                # swap gate
                elif op['name'] == 'swap':
                    self._swap(q_xy[0])
                    self._swap(q_xy[1])
                # add qubit-qubit wiring
                self._line(qreg_b, qreg_t)
            #
            # draw multi-qubit gates (n=3)
            #
            elif len(q_xy) == 3:
                # cswap gate
                if op['name'] == 'cswap':
                    self._ctrl_qubit(q_xy[0])
                    self._swap(q_xy[1])
                    self._swap(q_xy[2])
                # ccx gate
                elif op['name'] == 'ccx':
                    self._ctrl_qubit(q_xy[0])
                    self._ctrl_qubit(q_xy[1])
                    self._tgt_qubit(q_xy[2])
                # add qubit-qubit wiring
                self._line(qreg_b, qreg_t)
            else:
                logger.critical('Invalid gate %s', op)
                raise QISKitError('invalid gate {}'.format(op))
        #
        # adjust window size and draw horizontal lines
        #
        max_anc = max([q_anchors[ii].get_index() for ii in self._qreg_dict])
        n_fold = (max_anc - 1) // self._style.fold
        # window size
        if max_anc > self._style.fold > 0:
            self._cond['xmax'] = self._style.fold + 1
            self._cond['ymax'] = (n_fold + 1) * (self._cond['n_lines'] + 1) - 1
        else:
            self._cond['xmax'] = max_anc + 1
            self._cond['ymax'] = self._cond['n_lines']
        # add horizontal lines
        for ii in range(n_fold + 1):
            feedline_r = (n_fold > 0 and n_fold > ii)
            feedline_l = (ii > 0)
            self._draw_regs_sub(ii, feedline_l, feedline_r)
        # draw gate number
        if self._style.index:
            for ii in range(max_anc):
                if self._style.fold > 0:
                    x_coord = ii % self._style.fold + 1
                    y_coord = - (ii // self._style.fold) * (self._cond['n_lines'] + 1) + 0.7
                else:
                    x_coord = ii + 1
                    y_coord = 0.7
                self.ax.text(x_coord, y_coord, str(ii + 1), ha='center', va='center',
                             fontsize=self._style.sfs,
                             color=self._style.tc,
                             clip_on=True,
                             zorder=PORDER_TEXT)

    @staticmethod
    def param_parse(v, pimode=False):
        for i, e in enumerate(v):
            if pimode:
                v[i] = MatplotlibDrawer.format_pi(e)
            else:
                v[i] = MatplotlibDrawer.format_numeric(e)
            if v[i].startswith('-'):
                v[i] = '$-$' + v[i][1:]
        param = ', '.join(v)
        return param

    @staticmethod
    def format_pi(val):
        fracvals = MatplotlibDrawer.fraction(val)
        buf = ''
        if fracvals:
            nmr, dnm = fracvals.numerator, fracvals.denominator
            if nmr == 1:
                buf += '$\\pi$'
            elif nmr == -1:
                buf += '-$\\pi$'
            else:
                buf += '{}$\\pi$'.format(nmr)
            if dnm > 1:
                buf += '/{}'.format(dnm)
            return buf
        else:
            coef = MatplotlibDrawer.format_numeric(val / np.pi)
            if coef == '0':
                return '0'
            return '{}$\\pi$'.format(coef)

    @staticmethod
    def format_numeric(val, tol=1e-5):
        abs_val = abs(val)
        if isclose(abs_val, 0.0, abs_tol=1e-100):
            return '0'
        if isclose(fmod(abs_val, 1.0), 0.0, abs_tol=tol) and 0.5 < abs_val < 9999.5:
            return str(int(val))
        elif 0.1 <= abs_val < 100.0:
            return '{:.2f}'.format(val)
        return '{:.1e}'.format(val)

    @staticmethod
    def fraction(val, base=np.pi, n=100, tol=1e-5):
        abs_val = abs(val)
        for i in range(1, n):
            for j in range(1, n):
                if isclose(abs_val, i / j * base, rel_tol=tol):
                    if val < 0:
                        i *= -1
                    return Fraction(i, j)
        return None
