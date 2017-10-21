QISKit overview
===============

Philosophy
----------

QISKit is a collection of software for working with short depth
quantum circuits and building near term applications and experiments
on quantum computers. In QISKit, a quantum program is an array of
quantum circuits.  The program workflow consists of three stages:
Build, Compile, and Run. Build allows you to make different quantum
circuits that represent the problem you are solving. Compile allows
you to rewrite them to run on different backends (simulators/real
chips of different quantum volumes, sizes, fidelity, etc). Run
launches the jobs. After the jobs have been run, the data is
collected. There are methods for putting this data together, depending
on the program. This either gives you the answer you wanted or allows
you to make a better program for the next instance.

Project Overview
----------------
The QISKit project comprises:

* `QISKit SDK <https://github.com/QISKit/qiskit-sdk-py>`_: Python software 
  development kit for writing quantum computing experiments, programs, and 
  applications.

* `QISKit API <https://github.com/QISKit/qiskit-api-py>`_: A thin Python
  wrapper around the Quantum Experience HTTP API that enables you to
  connect and and execute quantum programs.

* `QISKit OpenQASM <https://github.com/QISKit/qiskit-openqasm>`_: Contains
  specifications, examples, documentation, and tools for the OpenQASM
  intermediate representation.

* `QISKit Tutorial <https://github.com/QISKit/qiskit-tutorial>`_: A 
  collection of Jupyter notebooks using QISKit.
