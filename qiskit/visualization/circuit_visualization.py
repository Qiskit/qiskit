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


"""
Module for the primary interface to the circuit drawers.

This module contains the end user facing API for drawing quantum circuits.
There are 3 available drawer backends available:

 0. ASCII art
 1. LaTeX
 2. Matplotlib

This provides a single function entry point to drawing a circuit object with
any of the backends.
"""

import errno
import logging
import os
import subprocess
import tempfile

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from qiskit import user_config
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization import latex as _latex
from qiskit.visualization import text as _text
from qiskit.visualization import utils
from qiskit.visualization import matplotlib as _matplotlib

logger = logging.getLogger(__name__)


def circuit_drawer(
    circuit,
    scale=None,
    filename=None,
    style=None,
    output=None,
    interactive=False,
    plot_barriers=True,
    reverse_bits=False,
    justify=None,
    vertical_compression="medium",
    idle_wires=True,
    with_layout=True,
    fold=None,
    ax=None,
    initial_state=False,
    cregbundle=True,
):
    """Draw the quantum circuit. Use the output parameter to choose the drawing format:

    **text**: ASCII art TextDrawing that can be printed in the console.

    **matplotlib**: images with color rendered purely in Python.

    **latex**: high-quality images compiled via latex.

    **latex_source**: raw uncompiled latex output.

    Args:
        circuit (QuantumCircuit): the quantum circuit to draw
        scale (float): scale of image to draw (shrink if < 1.0). Only used by
            the `mpl`, `latex` and `latex_source` outputs. Defaults to 1.0.
        filename (str): file path to save image to. Defaults to None.
        style (dict or str): dictionary of style or file name of style json file.
            This option is only used by the `mpl` or `latex` output type.
            If `style` is a str, it is used as the path to a json file
            which contains a style dict. The file will be opened, parsed, and
            then any style elements in the dict will replace the default values
            in the input dict. A file to be loaded must end in ``.json``, but
            the name entered here can omit ``.json``. For example,
            ``style='iqx.json'`` or ``style='iqx'``.
            If `style` is a dict and the ``'name'`` key is set, that name
            will be used to load a json file, followed by loading the other
            items in the style dict. For example, ``style={'name': 'iqx'}``.
            If `style` is not a str and `name` is not a key in the style dict,
            then the default value from the user config file (usually
            ``~/.qiskit/settings.conf``) will be used, for example,
            ``circuit_mpl_style = iqx``.
            If none of these are set, the `default` style will be used.
            The search path for style json files can be specified in the user
            config, for example,
            ``circuit_mpl_style_path = /home/user/styles:/home/user``.
            See: :class:`~qiskit.visualization.qcstyle.DefaultStyle` for more
            information on the contents.
        output (str): select the output method to use for drawing the circuit.
            Valid choices are ``text``, ``mpl``, ``latex``, ``latex_source``.
            By default the `text` drawer is used unless the user config file
            (usually ``~/.qiskit/settings.conf``) has an alternative backend set
            as the default. For example, ``circuit_drawer = latex``. If the output
            kwarg is set, that backend will always be used over the default in
            the user config file.
        interactive (bool): when set to true, show the circuit in a new window
            (for `mpl` this depends on the matplotlib backend being used
            supporting this). Note when used with either the `text` or the
            `latex_source` output type this has no effect and will be silently
            ignored. Defaults to False.
        reverse_bits (bool): when set to True, reverse the bit order inside
            registers for the output visualization. Defaults to False.
        plot_barriers (bool): enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (string): options are ``left``, ``right`` or ``none``. If
            anything else is supplied, it defaults to left justified. It refers
            to where gates should be placed in the output circuit if there is
            an option. ``none`` results in each gate being placed in its own
            column.
        vertical_compression (string): ``high``, ``medium`` or ``low``. It
            merges the lines generated by the `text` output so the drawing
            will take less vertical room.  Default is ``medium``. Only used by
            the `text` output, will be silently ignored otherwise.
        idle_wires (bool): include idle wires (wires with no circuit elements)
            in output visualization. Default is True.
        with_layout (bool): include layout information, with labels on the
            physical layout. Default is True.
        fold (int): sets pagination. It can be disabled using -1. In `text`,
            sets the length of the lines. This is useful when the drawing does
            not fit in the console. If None (default), it will try to guess the
            console width using ``shutil.get_terminal_size()``. However, if
            running in jupyter, the default line length is set to 80 characters.
            In `mpl`, it is the number of (visual) layers before folding.
            Default is 25.
        ax (matplotlib.axes.Axes): Only used by the `mpl` backend. An optional
            Axes object to be used for the visualization output. If none is
            specified, a new matplotlib Figure will be created and used.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        initial_state (bool): optional. Adds ``|0>`` in the beginning of the wire.
            Default is False.
        cregbundle (bool): optional. If set True, bundle classical registers.
            Default is True.

    Returns:
        :class:`TextDrawing` or :class:`matplotlib.figure` or :class:`PIL.Image` or
        :class:`str`:

        * `TextDrawing` (output='text')
            A drawing that can be printed as ascii art.
        * `matplotlib.figure.Figure` (output='mpl')
            A matplotlib figure object for the circuit diagram.
        * `PIL.Image` (output='latex')
            An in-memory representation of the image of the circuit diagram.
        * `str` (output='latex_source')
            The LaTeX source code for visualizing the circuit diagram.

    Raises:
        VisualizationError: when an invalid output method is selected
        MissingOptionalLibraryError: when the output methods requires non-installed libraries.

    Example:
        .. jupyter-execute::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.tools.visualization import circuit_drawer
            q = QuantumRegister(1)
            c = ClassicalRegister(1)
            qc = QuantumCircuit(q, c)
            qc.h(q)
            qc.measure(q, c)
            circuit_drawer(qc, output='mpl', style={'backgroundcolor': '#EEEEEE'})
    """
    image = None
    config = user_config.get_config()
    # Get default from config file else use text
    default_output = "text"
    if config:
        default_output = config.get("circuit_drawer", "text")
        if default_output == "auto":
            if _matplotlib.HAS_MATPLOTLIB:
                default_output = "mpl"
            else:
                default_output = "text"
    if output is None:
        output = default_output

    if output == "text":
        return _text_circuit_drawer(
            circuit,
            filename=filename,
            reverse_bits=reverse_bits,
            plot_barriers=plot_barriers,
            justify=justify,
            vertical_compression=vertical_compression,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            initial_state=initial_state,
            cregbundle=cregbundle,
        )
    elif output == "latex":
        image = _latex_circuit_drawer(
            circuit,
            filename=filename,
            scale=scale,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            initial_state=initial_state,
            cregbundle=cregbundle,
        )
    elif output == "latex_source":
        return _generate_latex_source(
            circuit,
            filename=filename,
            scale=scale,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            initial_state=initial_state,
            cregbundle=cregbundle,
        )
    elif output == "mpl":
        image = _matplotlib_circuit_drawer(
            circuit,
            scale=scale,
            filename=filename,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            ax=ax,
            initial_state=initial_state,
            cregbundle=cregbundle,
        )
    else:
        raise VisualizationError(
            "Invalid output type %s selected. The only valid choices "
            "are text, latex, latex_source, and mpl" % output
        )
    if image and interactive:
        image.show()
    return image


# -----------------------------------------------------------------------------
# _text_circuit_drawer
# -----------------------------------------------------------------------------


def _text_circuit_drawer(
    circuit,
    filename=None,
    reverse_bits=False,
    plot_barriers=True,
    justify=None,
    vertical_compression="high",
    idle_wires=True,
    with_layout=True,
    fold=None,
    initial_state=True,
    cregbundle=False,
    encoding=None,
):
    """Draws a circuit using ascii art.

    Args:
        circuit (QuantumCircuit): Input circuit
        filename (str): Optional filename to write the result
        reverse_bits (bool): Rearrange the bits in reverse order.
        plot_barriers (bool): Draws the barriers when they are there.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        vertical_compression (string): `high`, `medium`, or `low`. It merges the
            lines so the drawing will take less vertical room. Default is `high`.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information with labels on the physical
            layout. Default: True
        fold (int): Optional. Breaks the circuit drawing to this length. This
            is useful when the drawing does not fit in the console. If
            None (default), it will try to guess the console width using
            `shutil.get_terminal_size()`. If you don't want pagination
            at all, set `fold=-1`.
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True, bundle classical registers.
            Default: ``True``.
        encoding (str): Optional. Sets the encoding preference of the output.
            Default: ``sys.stdout.encoding``.

    Returns:
        TextDrawing: An instance that, when printed, draws the circuit in ascii art.
    """
    qubits, clbits, nodes = utils._get_layered_instructions(
        circuit, reverse_bits=reverse_bits, justify=justify, idle_wires=idle_wires
    )

    if with_layout:
        layout = circuit._layout
    else:
        layout = None
    global_phase = circuit.global_phase if hasattr(circuit, "global_phase") else None
    text_drawing = _text.TextDrawing(
        qubits,
        clbits,
        nodes,
        reverse_bits=reverse_bits,
        layout=layout,
        initial_state=initial_state,
        cregbundle=cregbundle,
        global_phase=global_phase,
        encoding=encoding,
        qregs=circuit.qregs,
        cregs=circuit.cregs,
    )
    text_drawing.plotbarriers = plot_barriers
    text_drawing.line_length = fold
    text_drawing.vertical_compression = vertical_compression

    if filename:
        text_drawing.dump(filename, encoding=encoding)
    return text_drawing


# -----------------------------------------------------------------------------
# latex_circuit_drawer
# -----------------------------------------------------------------------------


def _latex_circuit_drawer(
    circuit,
    scale=0.7,
    style=None,
    filename=None,
    plot_barriers=True,
    reverse_bits=False,
    justify=None,
    idle_wires=True,
    with_layout=True,
    initial_state=False,
    cregbundle=False,
):
    """Draw a quantum circuit based on latex (Qcircuit package)

    Requires version >=2.6.0 of the qcircuit LaTeX package.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        style (dict or str): dictionary of style or file name of style file
        filename (str): file path to save image to
        reverse_bits (bool): When set to True reverse the bit order inside
            registers for the output visualization.
        plot_barriers (bool): Enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information, with labels on the physical
            layout. Default: True
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True, bundle classical registers.
            Default: ``False``.

    Returns:
        PIL.Image: an in-memory representation of the circuit diagram

    Raises:
        OSError: usually indicates that ```pdflatex``` or ```pdftocairo``` is
                 missing.
        CalledProcessError: usually points to errors during diagram creation.
        MissingOptionalLibraryError: if pillow is not installed
    """
    tmpfilename = "circuit"
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, tmpfilename + ".tex")
        _generate_latex_source(
            circuit,
            filename=tmppath,
            scale=scale,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            initial_state=initial_state,
            cregbundle=cregbundle,
        )
        try:

            subprocess.run(
                [
                    "pdflatex",
                    "-halt-on-error",
                    f"-output-directory={tmpdirname}",
                    f"{tmpfilename + '.tex'}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except OSError as ex:
            if ex.errno == errno.ENOENT:
                logger.warning(
                    "WARNING: Unable to compile latex. "
                    "Is `pdflatex` installed? "
                    "Skipping latex circuit drawing..."
                )
            raise
        except subprocess.CalledProcessError as ex:
            with open("latex_error.log", "wb") as error_file:
                error_file.write(ex.stdout)
            logger.warning(
                "WARNING Unable to compile latex. "
                "The output from the pdflatex command can "
                "be found in latex_error.log"
            )
            raise
        else:
            if not HAS_PIL:
                raise MissingOptionalLibraryError(
                    libname="pillow",
                    name="latex drawer",
                    pip_install="pip install pillow",
                )
            try:
                base = os.path.join(tmpdirname, tmpfilename)
                subprocess.run(
                    ["pdftocairo", "-singlefile", "-png", "-q", base + ".pdf", base], check=True
                )
                image = Image.open(base + ".png")
                image = utils._trim(image)
                os.remove(base + ".png")
                if filename:
                    if filename.endswith(".pdf"):
                        os.rename(base + ".pdf", filename)
                    else:
                        image.save(filename, "PNG")
            except (OSError, subprocess.CalledProcessError) as ex:
                logger.warning(
                    "WARNING: Unable to convert pdf to image. "
                    "Is `poppler` installed? "
                    "Skipping circuit drawing..."
                )
                raise
        return image


def _generate_latex_source(
    circuit,
    filename=None,
    scale=0.7,
    style=None,
    reverse_bits=False,
    plot_barriers=True,
    justify=None,
    idle_wires=True,
    with_layout=True,
    initial_state=False,
    cregbundle=False,
):
    """Convert QuantumCircuit to LaTeX string.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        style (dict or str): dictionary of style or file name of style file
        filename (str): optional filename to write latex
        reverse_bits (bool): When set to True reverse the bit order inside
            registers for the output visualization.
        plot_barriers (bool): Enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information, with labels on the physical
            layout. Default: True
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True, bundle classical registers.
            Default: ``False``.

    Returns:
        str: Latex string appropriate for writing to file.
    """
    qubits, clbits, nodes = utils._get_layered_instructions(
        circuit, reverse_bits=reverse_bits, justify=justify, idle_wires=idle_wires
    )
    if with_layout:
        layout = circuit._layout
    else:
        layout = None

    global_phase = circuit.global_phase if hasattr(circuit, "global_phase") else None
    qcimg = _latex.QCircuitImage(
        qubits,
        clbits,
        nodes,
        scale,
        style=style,
        reverse_bits=reverse_bits,
        plot_barriers=plot_barriers,
        layout=layout,
        initial_state=initial_state,
        cregbundle=cregbundle,
        global_phase=global_phase,
        qregs=circuit.qregs,
        cregs=circuit.cregs,
    )
    latex = qcimg.latex()
    if filename:
        with open(filename, "w") as latex_file:
            latex_file.write(latex)

    return latex


# -----------------------------------------------------------------------------
# matplotlib_circuit_drawer
# -----------------------------------------------------------------------------


def _matplotlib_circuit_drawer(
    circuit,
    scale=None,
    filename=None,
    style=None,
    plot_barriers=True,
    reverse_bits=False,
    justify=None,
    idle_wires=True,
    with_layout=True,
    fold=None,
    ax=None,
    initial_state=False,
    cregbundle=True,
):

    """Draw a quantum circuit based on matplotlib.
    If `%matplotlib inline` is invoked in a Jupyter notebook, it visualizes a circuit inline.
    We recommend `%config InlineBackend.figure_format = 'svg'` for the inline visualization.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        filename (str): file path to save image to
        style (dict or str): dictionary of style or file name of style file
        reverse_bits (bool): When set to True, reverse the bit order inside
            registers for the output visualization.
        plot_barriers (bool): Enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (str): `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information, with labels on the physical
            layout. Default: True.
        fold (int): Number of vertical layers allowed before folding. Default is 25.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified, a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True bundle classical registers.
            Default: ``True``.

    Returns:
        matplotlib.figure: a matplotlib figure object for the circuit diagram
            if the ``ax`` kwarg is not set.
    """

    qubits, clbits, nodes = utils._get_layered_instructions(
        circuit, reverse_bits=reverse_bits, justify=justify, idle_wires=idle_wires
    )
    if with_layout:
        layout = circuit._layout
    else:
        layout = None

    if fold is None:
        fold = 25

    global_phase = circuit.global_phase if hasattr(circuit, "global_phase") else None
    qcd = _matplotlib.MatplotlibDrawer(
        qubits,
        clbits,
        nodes,
        scale=scale,
        style=style,
        reverse_bits=reverse_bits,
        plot_barriers=plot_barriers,
        layout=layout,
        fold=fold,
        ax=ax,
        initial_state=initial_state,
        cregbundle=cregbundle,
        global_phase=global_phase,
        qregs=circuit.qregs,
        cregs=circuit.cregs,
    )
    return qcd.draw(filename)
