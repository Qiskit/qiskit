{{ fullname }} package
{% for item in range(8 + fullname|length) -%}={%- endfor %}

.. automodule:: {{ fullname }}

{% if submodules %}
Submodules
----------

.. autosummary::
   :nosignatures:
   :toctree: ../_autodoc_public
{% for item in submodules %}
    {{ item }}
    {%- endfor %}
{%- endif %}


{% if subpackages %}
Subpackages
-----------

.. autosummary::
   :nosignatures:
   :toctree: ../_autodoc_public
{% for item in subpackages %}
    {{ item }}
    {%- endfor %}
{%- endif %}


{% if members_imports_refs %}
Contents
--------

.. autosummary::
    :nosignatures:
{% for item in members_imports_refs -%}
    {% set ref_type = item.split(':')[1] %}
    {% if ref_type != 'mod' -%}
    {% set ref_name = item.split(' ')[0].split('`')[1] -%}
    {{ ref_name }}
    {%- endif -%}
    {%- endfor -%}
{%- endif %}
