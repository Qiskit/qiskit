{%- if name.startswith('_') %}:orphan:{% endif %}


{{ fullname }} module
{% for item in range(7 + fullname|length) -%}={%- endfor %}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
    {% if members -%}
    :members: {{ members|join(", ") }}
    :undoc-members:
    :show-inheritance:

    Members
    -------

{%- if exceptions %}

    Exceptions
    ^^^^^^^^^^

    .. autosummary::
        :nosignatures:
{% for item in exceptions %}
        {{ item }}
{%- endfor %}
{%- endif %}

{%- if classes %}

    Classes
    ^^^^^^^

    .. autosummary::
        :nosignatures:
{% for item in classes %}
        {{ item }}
{%- endfor %}
{%- endif %}

{%- if functions %}

    Functions
    ^^^^^^^^^

    .. autosummary::
        :nosignatures:
{% for item in functions %}
        {{ item }}
{%- endfor %}
    {%- endif %}
{%- endif %}

{% set data = get_members(typ='data', in_list='__all__') %}
{%- if data %}

    Data
    ^^^^

    .. autosummary::
        :nosignatures:
{% for item in data %}
        {{ item }}
{%- endfor %}
    {%- endif %}

{% set all_refs = get_members(in_list='__all__', include_imported=True, out_format='refs') %}
{% if all_refs %}
    ``__all__``: {{ all_refs|join(", ") }}
{%- endif %}


{% if members %}
    Reference
    ---------

{%- endif %}
