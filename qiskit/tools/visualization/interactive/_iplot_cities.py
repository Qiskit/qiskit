# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Cities visualization
"""
from string import Template
import sys
import time
import re
from qiskit.tools.visualization._utils import _validate_input_state
if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def iplot_state_city(rho, figsize=None):
    """ Create a cities representation.

        Graphical representation of the input array using a city style graph.

        Args:
            rho (array): State vector or density matrix.
            figsize (tuple): The figure size in pixels.
    """

    # HTML
    html_template = Template("""
    <p>
        <div id="content_$divNumber" style="position: absolute; z-index: 1;">
            <div id="cities_$divNumber"></div>
        </div>
    </p>
    """)

    # JavaScript
    javascript_template = Template("""
    <script>
        requirejs.config({
            paths: {
                qVisualization: "https://qvisualization.mybluemix.net/q-visualizations"
            }
        });

        require(["qVisualization"], function(qVisualizations) {
            data = {
                real: $real,
                titleReal: "Real.[rho]",
                imaginary: $imag,
                titleImaginary: "Im.[rho]",
                qbits: $qbits
            };
            qVisualizations.plotState("cities_$divNumber",
                                      "cities",
                                      data,
                                      $options);
        });
    </script>
    """)
    rho = _validate_input_state(rho)
    if figsize is None:
        options = {}
    else:
        options = {'width': figsize[0], 'height': figsize[1]}
    # Process data and execute
    real = []
    imag = []
    for xvalue in rho:
        row_real = []
        col_imag = []

        for value_real in xvalue.real:
            row_real.append(float(value_real))
        real.append(row_real)

        for value_imag in xvalue.imag:
            col_imag.append(float(value_imag))
        imag.append(col_imag)

    div_number = str(time.time())
    div_number = re.sub('[.]', '', div_number)

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'real': real,
        'imag': imag,
        'qbits': len(real),
        'divNumber': div_number,
        'options': options
    })

    display(HTML(html + javascript))
