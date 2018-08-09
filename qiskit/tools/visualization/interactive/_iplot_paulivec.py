# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Paulivec visualization
"""
import numpy as np
from IPython.core.display import display, HTML
from string import Template
import time
import re
from qiskit.tools.qi.pauli import pauli_group


def process_data(rho):
    result = dict()

    num = int(np.log2(len(rho)))
    labels = list(map(lambda x: x.to_label(), pauli_group(num)))
    values = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                      pauli_group(num)))

    for position, label in enumerate(labels):
        result[label] = values[position]
    return result


def iplot_paulivec(executions_results, options={}):
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

    # Process data and execute
    divNumber = str(time.time())
    divNumber = re.sub('[.]', '', divNumber)

    if 'slider' in options and options['slider'] is True:
        options['slider'] = 1
    else:
        options['slider'] = 0

    if 'showLegend' in options and options['showLegend'] is False:
        options['showLegend'] = 0
    else:
        options['showLegend'] = 1

    data_to_plot = []
    for execution in executions_results:
        rho_data = process_data(execution['data'])
        rho_legend = execution['name']
        data_to_plot.append(dict(
            data=rho_data,
            name=rho_legend
        ))

    html = html_template.substitute({
        'divNumber': divNumber
    })

    javascript = javascript_template.substitute({
        'divNumber': divNumber,
        'executions': data_to_plot,
        'options': options
    })

    display(HTML(html + javascript))
