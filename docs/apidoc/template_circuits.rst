.. _qiskit-template-circuits:

.. automodule:: qiskit.circuit.library.templates
   :no-members:
   :no-inherited-members:
   :no-special-members:

Template Circuits
================

Templates are functions that return circuits that compute the identity. They are used in
circuit optimization where matching part of the template allows the compiler
to replace the match with the inverse of the remainder from the template.

NCT Template Circuits
--------------------

Template circuits for :class:`~qiskit.circuit.library.XGate`,
:class:`~qiskit.circuit.library.CXGate`, and
:class:`~qiskit.circuit.library.CCXGate` gates.

.. currentmodule:: qiskit.circuit.library.templates.nct
.. autofunction:: template_nct_2a_1
.. autofunction:: template_nct_2a_2
.. autofunction:: template_nct_2a_3
.. autofunction:: template_nct_4a_1
.. autofunction:: template_nct_4a_2
.. autofunction:: template_nct_4a_3
.. autofunction:: template_nct_4b_1
.. autofunction:: template_nct_4b_2
.. autofunction:: template_nct_5a_1
.. autofunction:: template_nct_5a_2
.. autofunction:: template_nct_5a_3
.. autofunction:: template_nct_5a_4
.. autofunction:: template_nct_6a_1
.. autofunction:: template_nct_6a_2
.. autofunction:: template_nct_6a_3
.. autofunction:: template_nct_6a_4
.. autofunction:: template_nct_6b_1
.. autofunction:: template_nct_6b_2
.. autofunction:: template_nct_6c_1
.. autofunction:: template_nct_7a_1
.. autofunction:: template_nct_7b_1
.. autofunction:: template_nct_7c_1
.. autofunction:: template_nct_7d_1
.. autofunction:: template_nct_7e_1
.. autofunction:: template_nct_9a_1
.. autofunction:: template_nct_9c_1
.. autofunction:: template_nct_9c_2
.. autofunction:: template_nct_9c_3
.. autofunction:: template_nct_9c_4
.. autofunction:: template_nct_9c_5
.. autofunction:: template_nct_9c_6
.. autofunction:: template_nct_9c_7
.. autofunction:: template_nct_9c_8
.. autofunction:: template_nct_9c_9
.. autofunction:: template_nct_9c_10
.. autofunction:: template_nct_9c_11
.. autofunction:: template_nct_9c_12
.. autofunction:: template_nct_9d_1
.. autofunction:: template_nct_9d_2
.. autofunction:: template_nct_9d_3
.. autofunction:: template_nct_9d_4
.. autofunction:: template_nct_9d_5
.. autofunction:: template_nct_9d_6
.. autofunction:: template_nct_9d_7
.. autofunction:: template_nct_9d_8
.. autofunction:: template_nct_9d_9
.. autofunction:: template_nct_9d_10

Clifford Template Circuits
--------------------------

Template circuits over Clifford gates.

.. currentmodule:: qiskit.circuit.library.templates.clifford
.. autofunction:: clifford_2_1
.. autofunction:: clifford_2_2
.. autofunction:: clifford_2_3
.. autofunction:: clifford_2_4
.. autofunction:: clifford_3_1
.. autofunction:: clifford_4_1
.. autofunction:: clifford_4_2
.. autofunction:: clifford_4_3
.. autofunction:: clifford_4_4
.. autofunction:: clifford_5_1
.. autofunction:: clifford_6_1
.. autofunction:: clifford_6_2
.. autofunction:: clifford_6_3
.. autofunction:: clifford_6_4
.. autofunction:: clifford_6_5
.. autofunction:: clifford_8_1
.. autofunction:: clifford_8_2
.. autofunction:: clifford_8_3

RZXGate Template Circuits
------------------------

Template circuits with :class:`~qiskit.circuit.library.RZXGate`.

.. currentmodule:: qiskit.circuit.library.templates.rzx
.. autofunction:: rzx_yz
.. autofunction:: rzx_xz
.. autofunction:: rzx_cy
.. autofunction:: rzx_zz1
.. autofunction:: rzx_zz2
.. autofunction:: rzx_zz3 