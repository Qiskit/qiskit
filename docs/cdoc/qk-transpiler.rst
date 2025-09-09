============
QkTranspiler
============

The :cpp:func:`qk_transpile` function lets C access the Qiskit tranpsiler (:py:mod:`qiskit.transpiler`).
The basic functionality is using the same underlying code as the Python-space version, but the transpiler as
exposed to C has more limitations than what is exposed to Python. The transpiler
assumes a circuit built constructed using solely the C API and is intended to
work solely in the case of a standalone C API. It will potentially not work
correctly when in a mixed Python/C use case. If you're mixing C and Python you
should call the :py:func:`.generate_preset_pass_manager` or
:py:func:`.transpile` functions for those circuits.

Data Types
==========

.. doxygenstruct:: QkTranspileResult
   :members:

.. doxygenstruct:: QkTranspileOptions
   :members:

Functions
=========

.. doxygengroup:: QkTranspiler
    :members:
    :content-only:
