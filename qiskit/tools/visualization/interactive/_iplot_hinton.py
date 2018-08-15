# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Hinton visualization
"""
from string import Template
import sys
import time
import re
if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def iplot_hinton(rho, options=None):
    """ Create a hinton representation.

        Graphical representation of the input array using a 2D city style
        graph (hinton).

        Args:
            rho (array): Density matrix
            options (dict): Representation settings containing
                    - width (integer): graph horizontal size
                    - height (integer): graph vertical size
    """

    # HTML
    html_template = Template("""
    <p>
        <div id="hinton_$divNumber"></div>
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
            qVisualizations.plotState("hinton_$divNumber",
                                      "hinton",
                                      $executions,
                                      $options);
        });
    </script>
    """)

    if not options:
        options = {}

    # Process data and execute
    div_number = str(time.time())
    div_number = re.sub('[.]', '', div_number)

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

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'divNumber': div_number,
        'executions': [{'data': real}, {'data': imag}],
        'options': options
    })

    display(HTML(html + javascript))
