# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Paulivec visualization
"""
from string import Template
import sys
import time
import re
import numpy as np
from qiskit.tools.qi.pauli import pauli_group
if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def process_data(rho):
    """ Sort rho data """
    result = dict()

    num = int(np.log2(len(rho)))
    labels = list(map(lambda x: x.to_label(), pauli_group(num)))
    values = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                      pauli_group(num)))

    for position, label in enumerate(labels):
        result[label] = values[position]
    return result


def iplot_paulivec(rho, options=None):
    """ Create a paulivec representation.

        Graphical representation of the input array.

        Args:
            rho (array): Density matrix
            options (dict): Representation settings containing
                    - width (integer): graph horizontal size
                    - height (integer): graph vertical size
                    - slider (bool): activate slider
                    - show_legend (bool): show legend of graph content
    """

    # HTML
    html_template = Template("""
    <p>
        <div id="paulivec_$divNumber"></div>
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
            qVisualizations.plotState("paulivec_$divNumber",
                                      "paulivec",
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

    if 'slider' in options and options['slider'] is True:
        options['slider'] = 1
    else:
        options['slider'] = 0

    if 'show_legend' in options and options['show_legend'] is False:
        options['show_legend'] = 0
    else:
        options['show_legend'] = 1

    data_to_plot = []
    rho_data = process_data(rho)
    data_to_plot.append(dict(
        data=rho_data
    ))

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'divNumber': div_number,
        'executions': data_to_plot,
        'options': options
    })

    display(HTML(html + javascript))
