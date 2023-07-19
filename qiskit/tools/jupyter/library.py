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

# pylint: disable=invalid-name,no-name-in-module,ungrouped-imports

"""A circuit library widget module"""

import ipywidgets as wid
from IPython.display import display
from qiskit import QuantumCircuit
from qiskit.utils import optionals as _optionals
from qiskit.utils.deprecation import deprecate_func


@_optionals.HAS_MATPLOTLIB.require_in_call
def _generate_circuit_library_visualization(circuit: QuantumCircuit):
    import matplotlib.pyplot as plt

    circuit = circuit.decompose()
    ops = circuit.count_ops()
    num_nl = circuit.num_nonlocal_gates()
    _fig, (ax0, ax1) = plt.subplots(2, 1)
    circuit.draw("mpl", ax=ax0)
    ax1.axis("off")
    ax1.grid(visible=None)
    ax1.table(
        [[circuit.name], [circuit.width()], [circuit.depth()], [sum(ops.values())], [num_nl]],
        rowLabels=["Circuit Name", "Width", "Depth", "Total Gates", "Non-local Gates"],
    )
    plt.tight_layout()
    plt.show()


@deprecate_func(
    since="0.25.0",
    additional_msg="This is unused by Qiskit, and no replacement will be publicly provided.",
)
def circuit_data_table(circuit: QuantumCircuit) -> wid.HTML:
    """Create a HTML table widget for a given quantum circuit.

    Args:
        circuit: Input quantum circuit.

    Returns:
        Output widget.
    """

    circuit = circuit.decompose()
    ops = circuit.count_ops()
    num_nl = circuit.num_nonlocal_gates()

    html = "<table>"
    html += """<style>
table {
    font-family: "IBM Plex Sans", Arial, Helvetica, sans-serif;
    border-collapse: collapse;
    width: 100%;
    border-left: 2px solid #212121;
}

th {
    text-align: left;
    padding: 5px 5px 5px 5px;
    width: 100%;
    background-color: #988AFC;
    color: #fff;
    font-size: 14px;
    border-left: 2px solid #988AFC;
}

td {
    text-align: left;
    padding: 5px 5px 5px 5px;
    width: 100%;
    font-size: 12px;
    font-weight: medium;
}

tr:nth-child(even) {background-color: #f6f6f6;}
</style>"""
    html += f"<tr><th>{circuit.name}</th><th></tr>"
    html += f"<tr><td>Width</td><td>{circuit.width()}</td></tr>"
    html += f"<tr><td>Depth</td><td>{circuit.depth()}</td></tr>"
    html += f"<tr><td>Total Gates</td><td>{sum(ops.values())}</td></tr>"
    html += f"<tr><td>Non-local Gates</td><td>{num_nl}</td></tr>"
    html += "</table>"

    out_wid = wid.HTML(html)
    return out_wid


head_style = (
    "font-family: IBM Plex Sans, Arial, Helvetica, sans-serif;"
    " font-size: 20px; font-weight: medium;"
)

property_label = wid.HTML(
    f"<p style='{head_style}'>Circuit Properties</p>",
    layout=wid.Layout(margin="0px 0px 10px 0px"),
)


@deprecate_func(
    since="0.25.0",
    additional_msg="This is unused by Qiskit, and no replacement will be publicly provided.",
)
def properties_widget(circuit: QuantumCircuit) -> wid.VBox:
    """Create a HTML table widget with header for a given quantum circuit.

    Args:
        circuit: Input quantum circuit.

    Returns:
        Output widget.
    """
    properties = wid.VBox(
        children=[property_label, circuit_data_table(circuit)],
        layout=wid.Layout(width="40%", height="auto"),
    )
    return properties


@_optionals.HAS_PYGMENTS.require_in_call
@deprecate_func(
    since="0.25.0",
    additional_msg="This is unused by Qiskit, and no replacement will be publicly provided.",
)
def qasm_widget(circuit: QuantumCircuit) -> wid.VBox:
    """Generate a QASM widget with header for a quantum circuit.

    Args:
        circuit: Input quantum circuit.

    Returns:
        Output widget.
    """
    import pygments
    from pygments.formatters import HtmlFormatter
    from qiskit.qasm.pygments import QasmHTMLStyle, OpenQASMLexer

    qasm_code = circuit.qasm()
    code = pygments.highlight(qasm_code, OpenQASMLexer(), HtmlFormatter())

    html_style = HtmlFormatter(style=QasmHTMLStyle).get_style_defs(".highlight")

    code_style = (
        """
    <style>
     .highlight
                {
                    font-family: monospace;
                    font-size: 14px;
                    line-height: 1.7em;
                }
     .highlight .err { color: #000000; background-color: #FFFFFF }
    %s
    </style>
    """
        % html_style
    )

    out = wid.HTML(
        code_style + code,
        layout=wid.Layout(max_height="500px", height="auto", overflow="scroll scroll"),
    )

    out_label = wid.HTML(
        f"<p style='{head_style}'>OpenQASM</p>",
        layout=wid.Layout(margin="0px 0px 10px 0px"),
    )

    qasm = wid.VBox(
        children=[out_label, out],
        layout=wid.Layout(
            height="auto", max_height="500px", width="60%", margin="0px 0px 0px 20px"
        ),
    )

    qasm._code_length = len(qasm_code.split("\n"))
    return qasm


@deprecate_func(
    since="0.25.0",
    additional_msg="This is unused by Qiskit, and no replacement will be publicly provided.",
)
def circuit_diagram_widget() -> wid.Box:
    """Create a circuit diagram widget.

    Returns:
        Output widget.
    """
    # The max circuit height corresponds to a 20Q circuit with flat
    # classical register.
    top_out = wid.Output(
        layout=wid.Layout(
            width="100%",
            height="auto",
            max_height="1000px",
            overflow="hidden scroll",
        )
    )

    top = wid.Box(children=[top_out], layout=wid.Layout(width="100%", height="auto"))

    return top


@deprecate_func(
    since="0.25.0",
    additional_msg="This is unused by Qiskit, and no replacement will be publicly provided.",
)
def circuit_library_widget(circuit: QuantumCircuit) -> None:
    """Create a circuit library widget.

    Args:
        circuit: Input quantum circuit.
    """
    qasm_wid = qasm_widget(circuit)
    sep_length = str(min(20 * qasm_wid._code_length, 495))

    # The separator widget
    sep = wid.HTML(
        f"<div style='border-left: 3px solid #212121;height: {sep_length}px;'></div>",
        layout=wid.Layout(height="auto", max_height="495px", margin="40px 0px 0px 20px"),
    )
    bottom = wid.HBox(
        children=[properties_widget(circuit), sep, qasm_widget(circuit)],
        layout=wid.Layout(max_height="550px", height="auto"),
    )

    top = circuit_diagram_widget()

    with top.children[0]:
        display(circuit.decompose().draw(output="mpl"))

    display(wid.VBox(children=[top, bottom], layout=wid.Layout(width="100%", height="auto")))
