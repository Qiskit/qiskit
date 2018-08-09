# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Hinton visualization
"""
from IPython.core.display import display, HTML
from string import Template
import time
import re


def iplot_hinton(executions_results, options={}):
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

    # Process data and execute
    divNumber = str(time.time())
    divNumber = re.sub('[.]', '', divNumber)

    # Process data and execute
    real = []
    imag = []
    for x in executions_results:
        row_real = []
        col_imag = []

        for value_real in x.real:
            row_real.append(float(value_real))
        real.append(row_real)

        for value_imag in x.imag:
            col_imag.append(float(value_imag))
        imag.append(col_imag)

    html = html_template.substitute({
        'divNumber': divNumber
    })

    javascript = javascript_template.substitute({
        'divNumber': divNumber,
        'executions': [dict(data=real), dict(data=imag)],
        'options': options
    })

    display(HTML(html + javascript))
