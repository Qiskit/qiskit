# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Histogram visualization
"""
from string import Template
from collections import Counter
import sys
import time
import re
import numpy as np
from .._error import VisualizationError

if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def process_data(data, number_to_keep):
    """ Prepare received data for representation.

        Args:
            data (dict): values to represent (ex. {'001' : 130})
            number_to_keep (int): number of elements to show individually.

        Returns:
            dict: processed data to show.
    """

    result = dict()

    if number_to_keep != 0:
        data_temp = dict(Counter(data).most_common(number_to_keep))
        data_temp['rest'] = sum(data.values()) - sum(data_temp.values())
        data = data_temp

    labels = data
    values = np.array([data[key] for key in labels], dtype=float)
    pvalues = values / sum(values)
    for position, label in enumerate(labels):
        result[label] = round(pvalues[position], 5)

    return result


def iplot_histogram(data, number_to_keep=False, options=None, legend=None):
    """ Create a histogram representation.

        Graphical representation of the input array using a vertical bars
        style graph.

        Args:
            data (list or dict):  This is either a list of dicts or a single
                dict containing the values to represent (ex. {'001' : 130})
            number_to_keep (int): DEPRECATED the number of terms to plot and
                rest is made into a single bar called other values
            legend (list): A list of strings to use for labels of the data.
                The number of entries must match the length of data.
            options (dict): Representation settings containing
                    - width (integer): graph horizontal size
                    - height (integer): graph vertical size
                    - slider (bool): activate slider
                    - number_to_keep (integer): groups max values
                    - show_legend (bool): show legend of graph content
                    - sort (string): Could be 'asc' or 'desc'
        Raises:
            VisualizationError: When legend is provided and the length doesn't
                match the input data.
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

    if 'show_legend' in options and options['show_legend'] is False:
        options['show_legend'] = 0
    else:
        options['show_legend'] = 1

    if 'number_to_keep' not in options:
        if number_to_keep is False:
            options['number_to_keep'] = 0
        elif number_to_keep:
            options['number_to_keep'] = number_to_keep

    data_to_plot = []
    if isinstance(data, dict):
        data = [data]

    if legend and len(legend) != len(data):
        raise VisualizationError("Length of legendL (%s) doesn't match number "
                                 "of input executions: %s" %
                                 (len(legend), len(data)))

    for item, execution in enumerate(data):
        exec_data = process_data(execution, options['number_to_keep'])
        out_dict = {'data': exec_data}
        if legend:
            out_dict['name'] = legend[item]
        data_to_plot.append(out_dict)

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'divNumber': div_number,
        'executions': data_to_plot,
        'options': options
    })

    display(HTML(html + javascript))
