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
    """ Create a histogram representation.

        Graphical representation of the input array using a vertical bars
        style graph.

        Args:
            executions_results (array): Array of dictionaries containing
                    - data (dict): values to represent (ex. {'001' : 130})
                    - name (string): name to show in the legend
                    - device (string): Could be 'real' or 'simulated'
            options (dict): Representation settings containing
                    - width (integer): graph horizontal size
                    - height (integer): graph vertical size
                    - slider (bool): activate slider
                    - rest (bool): make a group with all 0 value bars
                    - showLegend (bool): show legend of graph content
    """

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
