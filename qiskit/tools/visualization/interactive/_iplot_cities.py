# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Cities visualization
"""
from IPython.core.display import display, HTML
from string import Template
import time
import re


def iplot_cities(rho, options={}):
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

    # Process data and execute
    real = []
    imag = []
    for x in rho:
        row_real = []
        col_imag = []

        for value_real in x.real:
            row_real.append(float(value_real))
        real.append(row_real)

        for value_imag in x.imag:
            col_imag.append(float(value_imag))
        imag.append(col_imag)

    divNumber = str(time.time())
    divNumber = re.sub('[.]', '', divNumber)

    html = html_template.substitute({
        'divNumber': divNumber
    })

    javascript = javascript_template.substitute({
        'real': real,
        'imag': imag,
        'qbits': len(real),
        'divNumber': divNumber,
        'options': options
    })

    display(HTML(html + javascript))
