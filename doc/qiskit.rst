Qiskit overview
===============

Philosophy
----------

Qiskit is a collection of software for working with short-depth
quantum circuits, and running near-term applications and experiments
on quantum computers. In Qiskit, a quantum program is an array of
quantum circuits.  The program workflow consists of three stages:
Build, Compile, and Run. Build allows you to generate different quantum
circuits that represent the algorithm you are solving. Compile allows
you to rewrite them to run on different backends (simulators/real
chips of different quantum volumes, sizes, fidelity, etc). Run
launches the jobs. After the jobs have been run, the data is
collected. There are methods for putting this data together, depending
on the program. This either gives you the answer you wanted or allows
you to make a better program for the next instance.

Project Overview
----------------
The Qiskit project comprises:

* `Qiskit Terra <https://github.com/Qiskit/qiskit-terra>`_: Python science
  development kit for writing quantum computing experiments, programs, and
  applications.

* `Qiskit Aqua <https://github.com/Qiskit/aqua>`_:  A library and tools to 
  build applications for Noisy Intermediate-Scale Quantum (NISQ) computers

* `Qiskit API <https://github.com/Qiskit/qiskit-api-py>`_: A thin Python
  wrapper around the Quantum Experience HTTP API that enables you to
  connect and and execute quantum programs.

* `Qiskit OpenQASM <https://github.com/Qiskit/qiskit-openqasm>`_: Contains
  specifications, examples, documentation, and tools for the OpenQASM
  intermediate representation.

* `Qiskit Tutorial <https://github.com/Qiskit/qiskit-tutorial>`_: A
  collection of Jupyter notebooks using Qiskit.
