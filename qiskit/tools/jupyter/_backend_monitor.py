# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A module for monitoring backends."""

import math
import datetime
from IPython.display import display                     # pylint: disable=import-error
from IPython.core.magic import (line_magic,             # pylint: disable=import-error
                                Magics, magics_class)
import ipywidgets as widgets                            # pylint: disable=import-error
import matplotlib.pyplot as plt                         # pylint: disable=import-error
import matplotlib.colors                                # pylint: disable=import-error
import matplotlib as mpl                                # pylint: disable=import-error
from matplotlib import cm                               # pylint: disable=import-error
from matplotlib.patches import Circle                   # pylint: disable=import-error
from qiskit.providers.ibmq import IBMQ
from qiskit.qiskiterror import QISKitError
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.tools.visualization._gate_map import plot_gate_map


@magics_class
class BackendMonitor(Magics):
    """A class of status magic functions.
    """
    @line_magic
    def qiskit_backend_monitor(self, line='', cell=None):  # pylint: disable=W0613
        """A Jupyter magic function to monitor backends.
        """
        backend = self.shell.user_ns[line]
        if not isinstance(backend, IBMQBackend):
            raise QISKitError('Input variable is not of type IBMQBackend.')
        title_style = "style='color:#ffffff;background-color:#000000;padding-top: 1%;"
        title_style += "padding-bottom: 1%;padding-left: 1%; margin-top: 0px'"
        title_html = "<h1 {style}>{name}</h1>".format(
            style=title_style, name=backend.name())

        details = [config_tab(backend)]

        tab_contents = ['Configuration']

        if not backend.configuration().simulator:
            tab_contents.extend(['Qubit Properties', 'Multi-Qubit Gates',
                                 'Error Map', 'Job History'])
            details.extend([qubits_tab(backend), gates_tab(backend),
                            detailed_map(backend), job_history(backend)])

        tabs = widgets.Tab(layout=widgets.Layout(overflow_y='scroll'))
        tabs.children = details
        for i in range(len(details)):
            tabs.set_title(i, tab_contents[i])

        title_widget = widgets.HTML(value=title_html,
                                    layout=widgets.Layout(margin='0px 0px 0px 0px'))

        backend_monitor = widgets.VBox([title_widget, tabs],
                                       layout=widgets.Layout(border='4px solid #000000',
                                                             max_height='650px', min_height='650px',
                                                             overflow_y='hidden'))

        display(backend_monitor)


def config_tab(backend):
    """The backend configuration widget.

    Args:
        backend (IBMQbackend): The backend.

    Returns:
        grid: A GridBox widget.
    """
    status = backend.status().to_dict()
    config = backend.configuration().to_dict()

    config_dict = {**status, **config}

    upper_list = ['n_qubits', 'operational',
                  'status_msg', 'pending_jobs',
                  'basis_gates', 'local', 'simulator']

    lower_list = list(set(config_dict.keys()).difference(upper_list))
    # Remove gates because they are in a different tab
    lower_list.remove('gates')
    upper_str = "<table>"
    upper_str += """<style>
table {
    border-collapse: collapse;
    width: auto;
}

th, td {
    text-align: left;
    padding: 8px;
}

tr:nth-child(even) {background-color: #f6f6f6;}
</style>"""

    footer = "</table>"

    # Upper HBox widget data

    upper_str += "<tr><th>Property</th><th>Value</th></tr>"
    for key in upper_list:
        upper_str += "<tr><td><font style='font-weight:bold'>%s</font></td><td>%s</td></tr>" % (
            key, config_dict[key])
    upper_str += footer

    upper_table = widgets.HTML(
        value=upper_str, layout=widgets.Layout(width='100%', grid_area='left'))

    image_widget = widgets.Output(
        layout=widgets.Layout(display='flex-inline', grid_area='right',
                              padding='10px 10px 10px 10px',
                              width='auto', max_height='300px',
                              align_items='center'))

    if not config['simulator']:
        with image_widget:
            gate_map = plot_gate_map(backend)
            display(gate_map)
        plt.close(gate_map)

    lower_str = "<table>"
    lower_str += """<style>
table {
    border-collapse: collapse;
    width: auto;
}

th, td {
    text-align: left;
    padding: 8px;
}

tr:nth-child(even) {background-color: #f6f6f6;}
</style>"""
    lower_str += "<tr><th></th><th></th></tr>"
    for key in lower_list:
        if key != 'name':
            lower_str += "<tr><td>%s</td><td>%s</td></tr>" % (
                key, config_dict[key])
    lower_str += footer

    lower_table = widgets.HTML(value=lower_str,
                               layout=widgets.Layout(
                                   width='auto',
                                   grid_area='bottom'))

    grid = widgets.GridBox(children=[upper_table, image_widget, lower_table],
                           layout=widgets.Layout(
                               grid_template_rows='auto auto',
                               grid_template_columns='25% 25% 25% 25%',
                               grid_template_areas='''
                               "left right right right"
                               "bottom bottom bottom bottom"
                               ''',
                               grid_gap='0px 0px'))

    return grid


def qubits_tab(backend):
    """The qubits properties widget

    Args:
        backend (IBMQbackend): The backend.

    Returns:
        VBox: A VBox widget.
    """
    props = backend.properties().to_dict()

    header_html = "<div><font style='font-weight:bold'>{key}</font>: {value}</div>"
    header_html = header_html.format(key='last_update_date',
                                     value=props['last_update_date'])
    update_date_widget = widgets.HTML(value=header_html)

    qubit_html = "<table>"
    qubit_html += """<style>
table {
    border-collapse: collapse;
    width: auto;
}

th, td {
    text-align: left;
    padding: 8px;
}

tr:nth-child(even) {background-color: #f6f6f6;}
</style>"""

    qubit_html += "<tr><th></th><th>Frequency</th><th>T1</th><th>T2</th>"
    qubit_html += "<th>U1 gate error</th><th>U2 gate error</th><th>U3 gate error</th>"
    qubit_html += "<th>Readout error</th></tr>"
    qubit_footer = "</table>"

    for qub in range(len(props['qubits'])):
        name = 'Q%s' % qub
        qubit_data = props['qubits'][qub]
        gate_data = props['gates'][3*qub:3*qub+3]
        t1_info = qubit_data[0]
        t2_info = qubit_data[1]
        freq_info = qubit_data[2]
        readout_info = qubit_data[3]

        freq = str(round(freq_info['value'], 5))+' '+freq_info['unit']
        T1 = str(round(t1_info['value'],  # pylint: disable=invalid-name
                       5))+' ' + t1_info['unit']
        T2 = str(round(t2_info['value'],  # pylint: disable=invalid-name
                       5))+' ' + t2_info['unit']
        # pylint: disable=invalid-name
        U1 = str(round(gate_data[0]['parameters'][0]['value'], 5))
        # pylint: disable=invalid-name
        U2 = str(round(gate_data[1]['parameters'][0]['value'], 5))
        # pylint: disable=invalid-name
        U3 = str(round(gate_data[2]['parameters'][0]['value'], 5))

        readout_error = round(readout_info['value'], 5)
        qubit_html += "<tr><td><font style='font-weight:bold'>%s</font></td><td>%s</td>"
        qubit_html += "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        qubit_html = qubit_html % (name, freq, T1, T2, U1, U2, U3, readout_error)
    qubit_html += qubit_footer

    qubit_widget = widgets.HTML(value=qubit_html)

    out = widgets.VBox([update_date_widget,
                        qubit_widget])

    return out


def gates_tab(backend):
    """The multiple qubit gate error widget.

    Args:
        backend (IBMQbackend): The backend.

    Returns:
        VBox: A VBox widget.
    """
    config = backend.configuration().to_dict()
    props = backend.properties().to_dict()

    multi_qubit_gates = props['gates'][3*config['n_qubits']:]

    header_html = "<div><font style='font-weight:bold'>{key}</font>: {value}</div>"
    header_html = header_html.format(key='last_update_date',
                                     value=props['last_update_date'])

    update_date_widget = widgets.HTML(value=header_html,
                                      layout=widgets.Layout(grid_area='top'))

    gate_html = "<table>"
    gate_html += """<style>
table {
    border-collapse: collapse;
    width: auto;
}

th, td {
    text-align: left;
    padding: 8px;
}

tr:nth-child(even) {background-color: #f6f6f6;};
</style>"""

    gate_html += "<tr><th></th><th>Type</th><th>Gate error</th></tr>"
    gate_footer = "</table>"

    # Split gates into two columns
    left_num = math.ceil(len(multi_qubit_gates)/3)
    mid_num = math.ceil((len(multi_qubit_gates)-left_num)/2)

    left_table = gate_html

    for qub in range(left_num):
        gate = multi_qubit_gates[qub]
        name = gate['name']
        ttype = gate['gate']
        error = round(gate['parameters'][0]['value'], 5)

        left_table += "<tr><td><font style='font-weight:bold'>%s</font>"
        left_table += "</td><td>%s</td><td>%s</td></tr>"
        left_table = left_table % (name, ttype, error)
    left_table += gate_footer

    middle_table = gate_html

    for qub in range(left_num, left_num+mid_num):
        gate = multi_qubit_gates[qub]
        name = gate['name']
        ttype = gate['gate']
        error = round(gate['parameters'][0]['value'], 5)

        middle_table += "<tr><td><font style='font-weight:bold'>%s</font>"
        middle_table += "</td><td>%s</td><td>%s</td></tr>"
        middle_table = middle_table % (name, ttype, error)
    middle_table += gate_footer

    right_table = gate_html

    for qub in range(left_num+mid_num, len(multi_qubit_gates)):
        gate = multi_qubit_gates[qub]
        name = gate['name']
        ttype = gate['gate']
        error = round(gate['parameters'][0]['value'], 5)

        right_table += "<tr><td><font style='font-weight:bold'>%s</font>"
        right_table += "</td><td>%s</td><td>%s</td></tr>"
        right_table = right_table % (name, ttype, error)
    right_table += gate_footer

    left_table_widget = widgets.HTML(value=left_table,
                                     layout=widgets.Layout(grid_area='left'))
    middle_table_widget = widgets.HTML(value=middle_table,
                                       layout=widgets.Layout(grid_area='middle'))
    right_table_widget = widgets.HTML(value=right_table,
                                      layout=widgets.Layout(grid_area='right'))

    grid = widgets.GridBox(children=[update_date_widget,
                                     left_table_widget,
                                     middle_table_widget,
                                     right_table_widget],
                           layout=widgets.Layout(
                               grid_template_rows='auto auto',
                               grid_template_columns='33% 33% 33%',
                               grid_template_areas='''
                                                   "top top top"
                                                   "left middle right"
                                                   ''',
                               grid_gap='0px 0px'))

    return grid


def detailed_map(backend):
    """Widget for displaying detailed noise map.

    Args:
        backend (IBMQbackend): The backend.

    Returns:
        GridBox: Widget holding noise map images.
    """
    props = backend.properties().to_dict()
    config = backend.configuration().to_dict()
    single_gate_errors = [q['parameters'][0]['value']
                          for q in props['gates'][2:3*config['n_qubits']:3]]
    single_norm = matplotlib.colors.Normalize(
        vmin=min(single_gate_errors), vmax=max(single_gate_errors))
    q_colors = [cm.viridis(single_norm(err)) for err in single_gate_errors]

    cmap = config['coupling_map']

    cx_errors = []
    for line in cmap:
        for item in props['gates'][3*config['n_qubits']:]:
            if item['qubits'] == line:
                cx_errors.append(item['parameters'][0]['value'])
                break
        else:
            continue

    cx_norm = matplotlib.colors.Normalize(
        vmin=min(cx_errors), vmax=max(cx_errors))
    line_colors = [cm.viridis(cx_norm(err)) for err in cx_errors]

    single_widget = widgets.Output(layout=widgets.Layout(display='flex-inline', grid_area='left',
                                                         align_items='center'))

    cmap_widget = widgets.Output(layout=widgets.Layout(display='flex-inline', grid_area='top',
                                                       width='auto', height='auto',
                                                       align_items='center'))

    cx_widget = widgets.Output(layout=widgets.Layout(display='flex-inline', grid_area='right',
                                                     align_items='center'))

    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    with cmap_widget:
        noise_map = plot_gate_map(backend, qubit_color=q_colors,
                                  line_color=line_colors,
                                  qubit_size=28,
                                  plot_directed=True)
        width, height = noise_map.get_size_inches()

        noise_map.set_size_inches(1.25*width, 1.25*height)

        display(noise_map)
        plt.close(noise_map)

    with single_widget:
        cbl_fig = plt.figure(figsize=(3, 1))
        ax1 = cbl_fig.add_axes([0.05, 0.80, 0.9, 0.15])
        single_cb = mpl.colorbar.ColorbarBase(ax1, cmap=cm.viridis,
                                              norm=single_norm,
                                              orientation='horizontal')
        single_cb.locator = tick_locator
        single_cb.update_ticks()
        ax1.set_title('Single-qubit U3 error rate')
        display(cbl_fig)
        plt.close(cbl_fig)

    with cx_widget:
        cx_fig = plt.figure(figsize=(3, 1))
        ax2 = cx_fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cx_cb = mpl.colorbar.ColorbarBase(ax2, cmap=cm.viridis,
                                          norm=cx_norm,
                                          orientation='horizontal')
        cx_cb.locator = tick_locator
        cx_cb.update_ticks()
        ax2.set_title('CNOT error rate')
        display(cx_fig)
        plt.close(cx_fig)

    out_box = widgets.GridBox([single_widget, cmap_widget, cx_widget],
                              layout=widgets.Layout(
                                  grid_template_rows='auto auto',
                                  grid_template_columns='33% 33% 33%',
                                  grid_template_areas='''
                                                "top top top"
                                                "left . right"
                                                ''',
                                  grid_gap='0px 0px'))
    return out_box


def job_history(backend):
    """Widget for displaying job history

    Args:
     backend (IBMQbackend): The backend.

    Returns:
        Tab: A tab widget for history images.
    """
    year = widgets.Output(layout=widgets.Layout(display='flex-inline',
                                                align_items='center',
                                                min_height='400px'))

    month = widgets.Output(layout=widgets.Layout(display='flex-inline',
                                                 align_items='center',
                                                 min_height='400px'))

    week = widgets.Output(layout=widgets.Layout(display='flex-inline',
                                                align_items='center',
                                                min_height='400px'))

    tabs = widgets.Tab(layout=widgets.Layout(max_height='620px'))
    tabs.children = [year, month, week]
    tabs.set_title(0, 'Year')
    tabs.set_title(1, 'Month')
    tabs.set_title(2, 'Week')
    tabs.selected_index = 1

    _build_job_history(tabs, backend)
    return tabs


def _build_job_history(tabs, backend):

    backends = IBMQ.backends(backend.name())
    past_year_date = datetime.datetime.now() - datetime.timedelta(days=365)
    date_filter = {'creationDate': {'gt': past_year_date.isoformat()}}
    jobs = []
    for back in backends:
        jobs.extend(back.jobs(limit=None, db_filter=date_filter))

    with tabs.children[1]:
        month_plot = plot_job_history(jobs, interval='month')
        display(month_plot)
        plt.close(month_plot)

    with tabs.children[0]:
        year_plot = plot_job_history(jobs, interval='year')
        display(year_plot)
        plt.close(year_plot)

    with tabs.children[2]:
        week_plot = plot_job_history(jobs, interval='week')
        display(week_plot)
        plt.close(week_plot)


def plot_job_history(jobs, interval='year'):
    """Plots the job history of the user from the given list of jobs.

    Args:
        jobs (list): A list of jobs with type IBMQjob.
        interval (str): Interval over which to examine.

    Returns:
        fig: A Matplotlib figure instance.
    """
    def get_date(job):
        """Returns a datetime object from a IBMQJob instance.

        Args:
            job (IBMQJob): A job.

        Returns:
            dt: A datetime object.
        """
        return datetime.datetime.strptime(job.creation_date(),
                                          '%Y-%m-%dT%H:%M:%S.%fZ')

    current_time = datetime.datetime.now()

    if interval == 'year':
        bins = [(current_time - datetime.timedelta(days=k*365/12))
                for k in range(12)]
    elif interval == 'month':
        bins = [(current_time - datetime.timedelta(days=k)) for k in range(30)]
    elif interval == 'week':
        bins = [(current_time - datetime.timedelta(days=k)) for k in range(7)]

    binned_jobs = [0]*len(bins)

    if interval == 'year':
        for job in jobs:
            for ind, dat in enumerate(bins):
                date = get_date(job)
                if date.month == dat.month:
                    binned_jobs[ind] += 1
                    break
            else:
                continue
    else:
        for job in jobs:
            for ind, dat in enumerate(bins):
                date = get_date(job)
                if date.day == dat.day and date.month == dat.month:
                    binned_jobs[ind] += 1
                    break
            else:
                continue

    nz_bins = []
    nz_idx = []
    for ind, val in enumerate(binned_jobs):
        if val != 0:
            nz_idx.append(ind)
            nz_bins.append(val)

    total_jobs = sum(binned_jobs)

    colors = ['#003f5c', '#ffa600', '#374c80', '#ff764a',
              '#7a5195', '#ef5675', '#bc5090']

    if interval == 'year':
        labels = ['{}-{}'.format(str(bins[b].year)[2:], bins[b].month) for b in nz_idx]
    else:
        labels = ['{}-{}'.format(bins[b].month, bins[b].day) for b in nz_idx]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # pylint: disable=invalid-name
    ax.pie(nz_bins[::-1], labels=labels, colors=colors, textprops={'fontsize': 14},
           rotatelabels=True, counterclock=False)
    ax.add_artist(Circle((0, 0), 0.7, color='white', zorder=1))
    ax.text(0, 0, total_jobs, horizontalalignment='center',
            verticalalignment='center', fontsize=26)
    fig.tight_layout()
    return fig
