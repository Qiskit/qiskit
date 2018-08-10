# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Histogram visualization
"""
from string import Template
import time
import re
try:
    from IPython.core.display import display, HTML
except ImportError:
    print("Jupyter notebook is required")


def iplot_histogram(executions_results, options=None):
    """ Create a hinton representation """

    # HTML
    html_template = Template("""
    <p>
        <div id="histogram_$divNumber"></div>
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
            qVisualizations.plotState("histogram_$divNumber",
                                      "histogram",
                                      $executions,
                                      $options);
        });
    </script>
    """)

    # Process data and execute
    div_number = str(time.time())
    div_number = re.sub('[.]', '', div_number)

    if not options:
        options = {}

    if 'slider' in options and options['slider'] is True:
        options['slider'] = 1
    else:
        options['slider'] = 0

    if 'rest' in options and options['rest'] is True:
        options['rest'] = 1
    else:
        options['rest'] = 0

    if 'showLegend' in options and options['showLegend'] is False:
        options['showLegend'] = 0
    else:
        options['showLegend'] = 1

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'divNumber': div_number,
        'executions': executions_results,
        'options': options
    })

    display(HTML(html + javascript))
