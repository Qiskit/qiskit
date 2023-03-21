# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import custom_extensions
from custom_directives import (IncludeDirective, GalleryItemDirective,
                               CustomGalleryItemDirective, CustomCalloutItemDirective,
                               CustomCardItemDirective)


# -- General configuration ---------------------------------------------------

project = 'Qiskit'
copyright = f"2017-{datetime.date.today().year}, Qiskit Development Team"
author = 'Qiskit Development Team'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = '0.42.1'

rst_prolog = """
.. |version| replace:: {0}
""".format(release)

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx_autodoc_typehints',
    'jupyter_sphinx',
    'nbsphinx',
    'sphinx_design',
    'sphinx_reredirects',
    "matplotlib.sphinxext.plot_directive",
]

optimization_tutorials = [
    '1_quadratic_program',
    '2_converters_for_quadratic_programs',
    '3_minimum_eigen_optimizer',
    '4_grover_optimizer',
    '5_admm_optimizer',
    '6_examples_max_cut_and_tsp',
    '7_examples_vehicle_routing',
    '8_cvar_optimization',
    'index.html'
]

finance_tutorials = [
    '01_portfolio_optimization',
    '02_portfolio_diversification',
    '03_european_call_option_pricing',
    '04_european_put_option_pricing',
    '05_bull_spread_pricing',
    '06_basket_option_pricing',
    '07_asian_barrier_spread_pricing',
    '08_fixed_income_pricing',
    '09_credit_risk_analysis',
    '10_qgan_option_pricing',
    '11_time_series',
    'index'
]

chemistry_tutorials = [
    '01_electronic_structure',
    '02_vibronic_structure',
    '03_ground_state_solvers',
    '04_excited_states_solvers',
    '05_Sampling_potential_energy_surfaces',
    '06_calculating_thermodynamic_observables',
    'index'
]

ml_tutorials = [
    '01_qsvm_classification',
    '02_qsvm_multiclass',
    '03_vqc',
    '04_qgans_for_loading_random_distributions',
    'index'
]

dynamics_tutorials = [
    "09_pulse_simulator_duffing_model",
    "10_pulse_simulator_backend_model",
]

experiments_tutorials = [
    "1_hamiltonian_and_gate_characterization",
    "2_relaxation_and_decoherence",
    "3_measurement_error_mitigation",
    "4_randomized_benchmarking",
    "5_quantum_volume",
    "6_repetition_code",
    "7_accreditation",
    "8_tomography",
    "9_entanglement_verification",
    "index",
]


redirects = {
    "install": "getting_started.html",
}

for tutorial in optimization_tutorials:
    redirects['tutorials/optimization/%s' % tutorial] =  "https://qiskit.org/documentation/optimization/tutorials/index.html"

for tutorial in finance_tutorials:
    redirects['tutorials/finance/%s' % tutorial] = "https://qiskit.org/documentation/finance/tutorials/index.html"

for tutorial in chemistry_tutorials:
    redirects["tutorials/chemistry/%s" % tutorial] = "https://qiskit.org/documentation/nature/tutorials/index.html"

for tutorial in ml_tutorials:
    redirects["tutorials/machine_learning/%s" % tutorial] = "https://qiskit.org/documentation/machine-learning/tutorials/index.html"

for tutorial in dynamics_tutorials:
    redirects["tutorials/circuits_advanced/%s" % tutorial] = "https://qiskit.org/documentation/dynamics/tutorials/index.html"

for tutorial in experiments_tutorials:
    redirects["tutorials/noise/%s" % tutorial] = "https://qiskit.org/documentation/experiments/tutorials/index.html"

with open("aer_sources.txt", "r") as fd:
    for source_str in fd:
        target_str = f"../{source_str.replace('qiskit.providers.aer', 'qiskit_aer')}"
        redirects[source_str] = target_str

nbsphinx_timeout = 300
nbsphinx_execute = os.getenv('QISKIT_DOCS_BUILD_TUTORIALS', 'never')
nbsphinx_widgets_path = ''
html_sourcelink_suffix = ''
exclude_patterns = ['_build', '**.ipynb_checkpoints']

nbsphinx_thumbnails = {
    'tutorials/optimization/1_quadratic_program': 
    '_static/optimization/1_quadratic_program.png',
    'tutorials/optimization/2_converters_for_quadratic_programs': 
    '_static/optimization/2_converters.png',
    'tutorials/optimization/3_minimum_eigen_optimizer': 
    '_static/optimization/3_min_eig_opt.png',
    'tutorials/optimization/4_grover_optimizer': 
    '_static/optimization/4_grover.png',
    'tutorials/optimization/5_admm_optimizer': 
    '_static/optimization/5_ADMM.png',
}

nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html
    
    .. role:: raw-html(raw)
        :format: html
    
    .. note::
        This page was generated from `{{ docname }}`__.

    __ https://github.com/Qiskit/qiskit-tutorials/blob/master/{{ docname }}

"""

panels_css_variables = {
    "tabs-color-label-active": "rgb(138, 63, 252)",
    "tabs-color-label-inactive": "rgb(221, 225, 230)",
}
templates_path = ['_templates']

source_suffix = '.rst'
master_doc = 'index'

# Number figures, tables and code-blocks if they have a caption.
numfig = True
# Available keys are 'figure', 'table', 'code-block' and 'section'.  '%s' is    the number.
numfig_format = {'table': 'Table %s'}
# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# For Adding Locale
locale_dirs = ['locale/']   # path is example but recommended.
gettext_compact = False     # optional.

pygments_style = 'colorful'

# Whether module names are included in crossrefs of functions, classes, etc.
add_module_names = False

# A list of prefixes that are ignored for sorting the Python module index
# (e.g., if this is set to ['foo.'], then foo.bar is shown under B, not F).
# This can be handy if you document a project that consists of a single
# package. Works only for the HTML builder currently.
modindex_common_prefix = ['qiskit.']



# -- Configuration for extlinks extension ------------------------------------
# Refer to https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html
extlinks = {
    'pull_terra': ('https://github.com/Qiskit/qiskit-terra/pull/%s', '#'),
    'pull_aer': ('https://github.com/Qiskit/qiskit-aer/pull/%s', '#'),
    'pull_ibmq-provider': ('https://github.com/Qiskit/qiskit-ibmq-provider/pull/%s', '#')
}

# -- Options for HTML output -------------------------------------------------

html_theme = "qiskit_sphinx_theme"
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}
html_static_path = ['_static']
html_css_files = ['custom.css', 'gallery.css']
html_favicon = 'images/favicon.ico'
html_last_updated_fmt = '%Y/%m/%d'
html_context = {'analytics_enabled': os.getenv('QISKIT_ENABLE_ANALYTICS', False)} # enable segment analytics for qiskit.org/documentation

# -- Options for Autosummary and Autodoc ------------------------------------
# Note that setting autodoc defaults here may not have as much of an effect as  you may expect; any
# documentation created by autosummary uses a template file (in autosummary in  the templates path),
# which likely overrides the autodoc defaults.
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = 'both'

# -- Extension configuration -------------------------------------------------

def setup(app):
    app.add_directive('includenodoc', IncludeDirective)
    app.add_directive('galleryitem', GalleryItemDirective)
    app.add_directive('customgalleryitem', CustomGalleryItemDirective)
    app.add_directive('customcarditem', CustomCardItemDirective)
    app.add_directive('customcalloutitem', CustomCalloutItemDirective)
    custom_extensions.load_api_sources(app)
    custom_extensions.load_tutorials(app)
    app.setup_extension('versionutils')
    app.add_css_file('css/theme-override.css')
    app.connect('build-finished', custom_extensions.clean_api_source)
    app.connect('build-finished', custom_extensions.clean_tutorials)
    app.connect('source-read', custom_extensions.deprecate_ibmq_provider)
