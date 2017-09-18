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
{%- endfor -%}


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
   :toctree: ../_autodoc_public
{% for item in imported_classes %}
    {{ item }}
    {%- endfor %}

.. toctree::
    :hidden:
{% for item in imported_classes %}
    {{ fullname }}.{{ item }}
    {%- endfor %}
{%- endif %}

{% if imported_exceptions %}
Exceptions
----------

.. autosummary::
   :nosignatures:
   :toctree: ../_autodoc_public
{% for item in imported_exceptions %}
    {{ item }}
    {%- endfor %}

.. toctree::
    :hidden:
{% for item in imported_exceptions %}
    {{ fullname }}.{{ item }}
    {%- endfor %}
{%- endif %}

{% if imported_functions %}
Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../_autodoc_public
{% for item in imported_functions %}
    {{ item }}
    {%- endfor %}

.. toctree::
    :hidden:
{% for item in imported_functions %}
    {{ fullname }}.{{ item }}
    {%- endfor %}
{%- endif %}
