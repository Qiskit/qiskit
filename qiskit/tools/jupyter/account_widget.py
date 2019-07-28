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

"""A module of IBMQ account summary"""

import ipywidgets as widgets
from IPython.display import display


def _account_overview(account):
    """Grabs account information by HGP.

    Args:
        account (IBMQFactory): IBMQ account.

    Returns:
        dict: Account info.
    """
    providers = account.providers()

    hubs = {}
    for pro in providers:
        hub = pro.credentials.hub
        group = pro.credentials.group
        project = pro.credentials.project

        if hub not in hubs.keys():
            hubs[hub] = {}

        if group not in hubs[hub].keys():
            hubs[hub][group] = {}

        if project not in hubs[hub][group].keys():
            hubs[hub][group][project] = {'backends': None,
                                         'services': {'open_pulse': None,
                                                      'circuits': None}}

        backend_names = [back.name() for back in pro.backends()]
        open_pulse = [back.configuration().open_pulse for back in pro.backends()]
        hubs[hub][group][project]['backends'] = backend_names
        hubs[hub][group][project]['services']['open_pulse'] = open_pulse

    return hubs


def _project_widget(hgp_info, name):
    """Creates a widget for a single project.

    Args:
        hgp_info (dict): HGP information.
        name (str): The name to display.

    Returns:
        Widget: The project widget.
    """

    device_html = '<font size=2><b>Devices</b></font><br>'

    for back in hgp_info['backends']:
        device_html += '<code>%s</code><br>' % back

    devices_widget = widgets.HTML(device_html, layout=widgets.Layout(grid_area='names'))

    services_html = '<font size=2><b>Access Level</b></font><br>'

    device_ser = ['Gates']*len(hgp_info['backends'])

    for idx, op in enumerate(hgp_info['services']['open_pulse']):
        if op:
            device_ser[idx] += ', OpenPulse'

    for item in device_ser:
        services_html += item+'<br>'

    services_widget = widgets.HTML(services_html, layout=widgets.Layout(grid_area='access'))

    project_name = widgets.HTML("<font size=4>{name}</font>".format(name=name),
                                layout=widgets.Layout(grid_area='top'))

    items_widget = widgets.GridBox(children=[devices_widget, services_widget],
                                   layout=widgets.Layout(width='100%',
                                                         grid_template_columns='5% 30% 30%',
                                                         grid_template_rows='auto',
                                                         grid_template_areas='''
                                                        ". names access "
                                                            '''))

    project_widget = widgets.VBox(children=[project_name, items_widget])

    return project_widget


def _account_widget(account):
    """Build the IBMQ account widget.

    Args:
        account (IBMQFactory): IBMQ Account.
    """
    if not account.providers():
        return

    acct = _account_overview(account)

    title_style = "style='color:#ffffff;background-color:#000000;padding-top: 1%;"
    title_style += "padding-bottom: 1%;padding-left: 1%; margin-top: 0px'"
    title_html = "<h1 {style}>{name}</h1>".format(style=title_style, name='IBMQ Account')

    title_widget = widgets.HTML(title_html, layout=widgets.Layout(width='99%',
                                                                  margin='0px 0px 0px 0px'))

    hub_widgets = []
    for hub in acct:
        projects = []
        for group in acct[hub]:
            for project in acct[hub][group]:
                name = "{hub} / {group} / {project}".format(hub=hub, group=group, project=project)
                projects.append(_project_widget(acct[hub][group][project], name))

        hub_widgets.append(widgets.VBox(projects))

    hubs = widgets.Accordion(children=hub_widgets, layout=widgets.Layout(width='99%',
                                                                         max_height='600px',
                                                                         overflow_y='scroll'))

    for idx, hub in enumerate(acct.keys()):
        hubs.set_title(idx, hub)
    # If more than one hub, start with all closed
    if len(hub_widgets) > 1:
        hubs.selected_index = None

    acct_widget = widgets.VBox(children=[title_widget, hubs])

    display(acct_widget)
