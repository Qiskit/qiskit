{#
   We show all the class's methods and attributes on the same page. By default, we document
   all methods, including those defined by parent classes.
-#}

{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :show-inheritance:

{% block attributes_summary %}
  {% if attributes %}
   .. rubric:: Attributes
    {% for item in attributes %}
   .. autoattribute:: {{ item }}
    {%- endfor %}
  {% endif %}
{% endblock -%}

{% block methods_summary %}
  {% set wanted_methods = (methods | reject('==', '__init__') | list) %}
  {% if wanted_methods %}
   .. rubric:: Methods
    {% for item in wanted_methods %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
{% endblock %}
