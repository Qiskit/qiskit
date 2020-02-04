# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Interactive histogram from experiment counts"""

import functools
import numpy as np
from qiskit.exceptions import QiskitError
from ..counts_visualization import (_plot_histogram_data,
                                    VALID_SORTS, DIST_MEAS)


def iplot_histogram(data, figsize=(None, None), color=None,
                    number_to_keep=None,
                    sort='asc', target_string=None,
                    legend=None, bar_labels=True,
                    title=None, background_color='white',
                    as_widget=False):
    """Interactive histogram plot of counts data.

    Args:
        data (list or dict): This is either a list of dictionaries or a single
            dict containing the values to represent (ex {'001': 130})
        figsize (tuple): Figure size in pixels.
        color (list or str): String or list of strings for histogram bar colors.
        number_to_keep (int): The number of terms to plot and rest
            is made into a single bar called 'rest'.
        sort (string): Could be 'asc', 'desc', or 'hamming'.
        target_string (str): Target string if 'sort' is a distance measure.
        legend(list): A list of strings to use for labels of the data.
            The number of entries must match the length of data (if data is a
            list or 1 if it's a dict)
        bar_labels (bool): Label each bar in histogram with probability value.
        title (str): A string to use for the plot title.
        background_color (str): Set the background color to 'white'
                                or 'black'.
        as_widget (bool): Return figure as an ipywidget.

    Returns:
        PlotlyFigure or PlotlyWidget:
            A figure for the rendered histogram.

    Raises:
        ValueError: When legend is provided and the length doesn't
                    match the input data.
        QiskitError: When failed to load plotly.

    Example:
        .. jupyter-execute::

           from qiskit import QuantumCircuit, BasicAer, execute
           from theia.visualization import iplot_histogram

           qc = QuantumCircuit(2, 2)
           qc.h(0)
           qc.cx(0, 1)
           qc.measure([0, 1], [0, 1])

           backend = BasicAer.get_backend('qasm_simulator')
           job = execute(qc, backend)
           iplot_histogram(job.result().get_counts(), as_widget=True)
    """
    try:
        import plotly.graph_objects as go
        from .plotly_wrapper import PlotlyWidget, PlotlyFigure
    except ImportError:
        raise QiskitError('iplot_histogram requires plotly to be installed.')

    if sort not in VALID_SORTS:
        raise ValueError("Value of sort option, %s, isn't a "
                         "valid choice. Must be 'asc', "
                         "'desc', or 'hamming'")
    if sort in DIST_MEAS.keys() and target_string is None:
        err_msg = 'Must define target_string when using distance measure.'
        raise ValueError(err_msg)

    if isinstance(data, dict):
        data = [data]

    if legend and len(legend) != len(data):
        raise ValueError("Length of legendL (%s) doesn't match "
                         "number of input executions: %s" %
                         (len(legend), len(data)))

    if background_color == 'white':
        text_color = 'black'
    elif background_color == 'black':
        text_color = 'white'
    else:
        raise ValueError('Invalid background_color selection.')

    labels = list(sorted(
        functools.reduce(lambda x, y: x.union(y.keys()), data, set())))
    if number_to_keep is not None:
        labels.append('rest')

    if sort in DIST_MEAS.keys():
        dist = []
        for item in labels:
            dist.append(DIST_MEAS[sort](item, target_string))

        labels = [list(x) for x in zip(*sorted(zip(dist, labels),
                                               key=lambda pair: pair[0]))][1]

    # Set bar colors
    if color is None:
        color = ["#003f5c", "#ffa600", "#bc5090", "#58508d", "#ff6361"]
    elif isinstance(color, str):
        color = [color]

    width = 1/(len(data)+1)  # the width of the bars

    labels_dict, all_pvalues, _ = _plot_histogram_data(data,
                                                       labels,
                                                       number_to_keep)

    fig = go.Figure()
    for item, _ in enumerate(data):
        xvals = []
        yvals = []
        for idx, val in enumerate(all_pvalues[item]):
            xvals.append(idx+item*width)
            yvals.append(val)

        labels = list(labels_dict.keys())
        hover_template = "<b>{x}</b><br>P = {y}"
        hover_text = [hover_template.format(x=labels[kk],
                                            y=np.round(yvals[kk], 3)) for kk in range(len(yvals))]

        fig.add_trace(go.Bar(x=xvals,
                             y=yvals,
                             width=width,
                             hoverinfo="text",
                             hovertext=hover_text,
                             marker_color=color[item % len(color)],
                             name=legend[item] if legend else '',
                             text=np.round(yvals, 3) if bar_labels else None,
                             textposition='auto'
                             ))

    fig.update_xaxes(tickvals=list(range(len(labels_dict.keys()))),
                     ticktext=list(labels_dict.keys()),
                     tickfont_size=14,
                     showline=True, linewidth=1,
                     linecolor=text_color if text_color == 'white' else None,
                     )

    fig.update_yaxes(title='Probability',
                     titlefont_size=18,
                     tickfont_size=14,
                     showline=True, linewidth=1,
                     linecolor=text_color if text_color == 'white' else None,
                     )

    fig.update_layout(xaxis_tickangle=-70,
                      showlegend=(legend is not None),
                      width=figsize[0],
                      height=figsize[1],
                      paper_bgcolor=background_color,
                      margin=dict(t=40, l=50, r=10, b=10),
                      title=dict(text=title, x=0.5),
                      title_font_size=20,
                      font=dict(color=text_color),
                      )
    if as_widget:
        return PlotlyWidget(fig)

    return PlotlyFigure(fig)
