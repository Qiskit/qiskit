{{ fullname }} package
{% for item in range(8 + fullname|length) -%}={%- endfor %}

{# Split the imported references into several lists. #}
{% set all = get_members(in_list='__all__', include_imported=True) %}
{%- set exceptions = get_members(typ='exception', include_imported=True, out_format='table') -%}
{%- set classes = get_members(typ='class', include_imported=True, out_format='table') -%}
{%- set functions = get_members(typ='function', include_imported=True, out_format='table') -%}
{%- set data = get_members(typ='data', include_imported=True, out_format='table') -%}

.. automodule:: {{ fullname }}
    {% if members -%}
    :members: {{ members|join(", ") }}
    :undoc-members:
    :show-inheritance:
    {%- endif %}
    {% if fullname.startswith('_') %}
    :toctree: qiskit.private.{{ fullname }}
    {%- endif -%}

{% if submodules %}
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   {# Do not show the "private" (starting with `_`) modules. This workaround is
   due to the fact that we need the members in private modules parsed and
   generated but not show everything in the package (for example,
   `qiskit._foo.Bar` is shown as `Bar` in the `qiskit` package page, but
   `qiskit._foo` should not be linked directly or shown in the `qiskit` package
   page).
   See also the :orphaned: at the top, and the --private flag for
   better-apidoc. #}

{% for item in submodules %}
       {% if not item.startswith('_') %}{{ fullname }}.{{ item }}{%- endif -%}
       {%- endfor %}
{%- endif -%}

{% if subpackages %}

    Subpackages
    -----------

    .. toctree::
       :maxdepth: 1
{% for item in subpackages %}
       {{ fullname }}.{{ item }}
       {%- endfor %}
{%- endif %}

{%- if exceptions %}

    Exceptions
    ----------

{% for line in exceptions %}
    {{ line }}
{%- endfor %}
{%- endif %}

{%- if classes %}

    Classes
    -------

{% for line in classes %}
    {{ line }}
{%- endfor %}
{%- endif %}

{%- if functions %}

{# Manually name this section via a "_qiskit_top_level_functions" reference,
   for convenience (link from release notes). #}
{% if fullname == 'qiskit' %}
    .. _qiskit_top_level_functions:
{% endif %}

    Functions
    ---------

{% for line in functions %}
    {{ line }}
{%- endfor %}
{%- endif %}
