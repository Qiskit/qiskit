===========
QkComplex64
===========

A complex, double-precision number representation. This data type is used to safely 
use complex numbers within Qiskit, but is not necessarily designed for easy manipulation
on user-side. Instead, we provide functions to convert from compiler-native representation
(e.g. ``complex double`` for GNU or Clang compilers), which allow for ergonomic handling,
to Qiskit's ``QkComplex64`` representation, which is meant for passing into Qiskit functions 
and structs.

----------
Data types
----------

.. doxygenstruct:: QkComplex64
   :members:
   :undoc-members:


---------
Functions
---------

.. doxygengroup:: QkComplex64
   :content-only:
