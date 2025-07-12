=========================
QkElidePermutationsResult
=========================

.. code-block:: c

   typedef struct QkElidePermutationsResult QkElidePermutationsResult

When running the ``qk_transpiler_pass_standalone_elide_permutations`` function it returns a
modified circuit and a permutation array as a QkElidePermutationsResult object. This object
contains the outcome of the transpiler pass, whether the pass was able to elide any gates or not,
and what the results of that elision were.

Functions
=========

.. doxygengroup:: QkElidePermutationsResult
   :members:
   :content-only:
