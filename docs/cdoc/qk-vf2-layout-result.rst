=================
QkVF2LayoutResult
=================

.. code-block:: c

   typedef struct QkVF2LayoutResult QkVF2LayoutResult

When running the ``qk_transpiler_pass_standalone_vf2_layout`` function it returns its analysis
result as a ``QkVF2LayoutResult`` object. This object contains the outcome of the transpiler pass,
whether the pass was able to find a layout or not, and what the layout selected by the pass was.

Functions
=========

.. doxygengroup:: QkVF2LayoutResult
   :members:
   :content-only:
