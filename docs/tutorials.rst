.. _tutorials:

=========
Tutorials
=========

.. note::
  The Simulators tutorials have moved to
  `Qiskit Aer <https://qiskit.org/ecosystem/aer/tutorials/index.html>`_
  and the Algorithms tutorials to
  `Qiskit Algorithms <https://qiskit.org/ecosystem/algorithms/tutorials/index.html>`_.

Introductory
============

.. qiskit-card::
   :header: Qiskit warmup
   :card_description: An introduction to Qiskit and the primary user workflow.
   :image: _static/images/logo.png
   :link: get_started_with_qiskit.html

Quantum circuits
================

.. nbgallery::
   :glob:

   tutorials/circuits/*

Advanced circuits
=================

.. nbgallery::
   :glob:

   tutorials/circuits_advanced/*

Operators
=========

.. deprecated:: 0.24.0
   The operators tutorials rely on the ``opflow`` module, which has been deprecated since
   Qiskit 0.43 (aka Qiskit Terra 0.24). Refer to the
   `Opflow migration guide <https://qisk.it/opflow_migration>`_.
   These tutorials will be removed in the future.

.. nbgallery::
   :glob:

   tutorials/operators/*
