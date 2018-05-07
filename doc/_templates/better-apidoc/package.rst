{{ fullname }} package
{% for item in range(8 + fullname|length) -%}={%- endfor %}

.. automodule:: {{ fullname }}

{# Split the imported references into several lists, as better-apidoc seems to
   have a bug with our current public vs private structure and the variables
   provided by the extension are not always fully populated. #}
{%- set imported_modules = [] -%}
{%- set imported_classes = [] -%}
{%- set imported_exceptions = [] -%}
{%- set imported_functions = [] -%}
{%- set imported_other = [] -%}

{% for item in members_imports_refs -%}
    {%- if item.split('<')[1].split('>')[0].startswith('qiskit') -%}
        {%- set ref_type = item.split(':')[1] -%}
        {%- set ref_name = item.split(' ')[0].split('`')[1] -%}
        {%- if ref_type == 'mod' -%}
            {%- if ref_name != fullname and fullname != 'qiskit.extensions.standard' -%}
                {{- imported_modules.append(ref_name) or '' -}}
            {%- endif %}
        {%- elif ref_type == 'class' -%}
            {{- imported_classes.append(ref_name) or '' -}}
        {%- elif ref_type == 'exc' -%}
            {{- imported_exceptions.append(ref_name) or '' -}}
        {%- elif ref_type == 'func' -%}
            {{- imported_functions.append(ref_name) or '' -}}
        {%- else -%}
            {{- imported_other.append(ref_name) or '' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}

{# Bypass the automatic discovery of simulators at qiskit.backends and of
   gates. #}
{%- if fullname == 'qiskit.backends' -%}
    {%- set imported_modules = ['_qasm_simulator_cpp',
                                '_qasm_simulator_py',
                                '_unitary_simulator_py',
                                '_unitary_simulator_sympy',
                                '_statevector_simulator_cpp',
                                '_statevector_simulator_py',
                                '_statevector_simulator_sympy',
                                '_qasm_simulator_projectq'] -%}
    {%- set imported_classes = ['BaseBackend'] -%}
{%- elif fullname == 'qiskit.extensions' -%}
    {%- set imported_modules = ['standard',
                                'qasm_simulator_cpp',
                                'quantum_initializer'] -%}
{%- endif -%}

{% if imported_modules %}
Submodules
----------

.. autosummary::
   :nosignatures:
   :toctree:
{% for item in imported_modules %}
    {{ item }}
    {%- endfor %}
{%- endif %}

{% if imported_classes %}
Classes
-------

.. autosummary::
   :nosignatures:
   :toctree:
   :template: autosummary/class.rst
{% for item in imported_classes %}
    {{ item }}
{%- endfor %}
{%- endif %}

{% if imported_exceptions %}
Exceptions
----------

.. autosummary::
   :nosignatures:
   :toctree:
{% for item in imported_exceptions %}
    {{ item }}
{%- endfor %}
{%- endif %}

{% if imported_functions %}
{# Manually name this section via a "_qiskit_top_level_functions" reference,
   for convenience (link from release notes). #}
{% if fullname == 'qiskit' %}
.. _qiskit_top_level_functions:
{% endif %}

Functions
---------

.. autosummary::
   :nosignatures:
   {% if fullname != 'qiskit.extensions.standard' -%}:toctree:{% endif %}
{% for item in imported_functions %}
    {{ item }}
{%- endfor %}

{# Handle the qiskit.extensions.standard module, as the imports are in the form
   "from .ABC import ABC" except in two cases, which makes the documentation
   try to point to the submodules and not the actual functions. #}
{% if fullname == 'qiskit.extensions.standard' -%}
{%- for item in imported_functions %}
.. autofunction:: {{ item }}
{%- endfor %}
{%- endif %}

{%- endif %}
