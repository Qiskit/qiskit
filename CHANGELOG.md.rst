Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`_ and this project adheres to
`Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

   **Tags:**
   - 🎉 Added
   - ✏️ Changed
   - ⚠️ Deprecated
   - ❌ Removed
   - 🐛 Fixed
   - 👾 Security

`Unreleased`_
------------

✏️ Changed
~~~~~~~~~~

-  New Changelog based on `Keep a Changelog`_.
-  Rename repository to ``qiskit-core``.
-  Update the issue templates.

`0.5.3`_ - 2018-05-29
--------------------

🎉 Added
~~~~~~~

-  load_qasm_file / load_qasm_string methods

.. _changed-1:

✏️ Changed
~~~~~~~~~~

-  Dependencies version bumped

🐛 Fixed
~~~~~~~

-  Crash in the cpp simulator for some linux platforms
-  Fixed some minor bugs

.. _section-1:

`0.5.2`_ - 2018-05-21
--------------------

.. _changed-2:

✏️ Changed
~~~~~~~~~~

-  Adding Result.get_unitary()

⚠️ Deprecated
~~~~~~~~~~~~~

-  Deprecating ibmqx_hpc_qasm_simulator and ibmqx_qasm_simulator in
   favor of: ibmq_qasm_simulator.

.. _fixed-1:

🐛 Fixed
~~~~~~~

-  Fixing a Mapper issue.
-  Fixing Windows 7 builds.

.. _section-2:

`0.5.1`_ - 2018-05-15
--------------------

There are no code changes.

MacOS simulator has been rebuilt with external user libraries compiled
statically, so there’s no need for users to have a preinstalled gcc
environment.

Pypi forces us to bump up the version number if we want to upload a new
package, so this is basically what have changed.

.. _section-3:

`0.5.0`_ - 2018-05-11
--------------------

⚠️ TODO ⚠️

.. _section-4:

`0.4.15`_ - 2018-05-07
---------------------

.. _fixed-2:

🐛 Fixed
~~~~~~~

-  Fixed an issue with legacy code that was affecting Developers
   Challenge

.. _section-5:

`0.4.14`_ - 2018-04-18
---------------------

.. _fixed-3:

🐛 Fixed
~~~~~~~

-  Fixed an issue about handling Basis Gates parameters on backend
   configurations

.. _section-6:

`0.4.13`_ - 2018-04-16
---------------------

.. _changed-3:

✏️ Changed
~~~~~~~~~~

-  OpenQuantumCompiler.dag2json() restored for backward compatibility

.. _fixed-4:

🐛 Fixed
~~~~~~~

-  Fixes an issue regarding barrier gate misuse in some circumstances

.. _section-7:

`0.4.12`_ - 2018-03-11
---------------------

.. _changed-4:

✏️ Changed
~~~~~~~~~~

-  Improved circuit visualization.
-  Improvements in infrastructure code, mostly tests and build system.
-  Better documentation regarding contributors

.. _fixed-5:

🐛 Fixed
~~~~~~~

-  A bunch of minor bugs have been fixed.

.. _section-8:

`0.4.11`_ - 2018-03-13
---------------------

.. _added-1:

🎉 Added
~~~~~~~

-  More testing :)

.. _changed-5:

✏️ Changed
~~~~~~~~~~

-  Stabilizing code related to external dependencies

.. _fixed-6:

🐛 Fixed
~~~~~~~

-  Fixed bug in circuit drawing where some gates in the standard library
   were not plotting correctly

.. _section-9:

`0.4.10`_ - 2018-03-06
---------------------

.. _added-2:

🎉 Added
~~~~~~~

-  Chinese translation of README

.. _changed-6:

✏️ Changed
~~~~~~~~~~

-  Changes related with infrastructure (linter, tests, automation)
   enhancement

.. _fixed-7:

🐛 Fixed
~~~~~~~

-  Fix installation issue when simulator cannot be built
-  Fix bug with auto-generated CNOT coherent error matrix in C++
   simulator
-  Fix a bug in the async code

.. _section-10:

`0.4.9`_ - 2018-02-12
--------------------

.. _changed-7:

✏️ Changed
~~~~~~~~~~

-  CMake integration
-  QASM improvements
-  Mapper optimizer improvements

.. _fixed-8:

🐛 Fixed
~~~~~~~

-  Some minor C++ Simulator bug-fixes

.. _section-11:

`0.4.8`_ - 2018-01-29
--------------------

.. _fixed-9:

🐛 Fixed
~~~~~~~

-  Fix parsing U_error matrix in C++ Simulator python helper class
-  Fix display of code-blocks on .rst pages

`0.4.7`_ - 2018-01-26
---------------------

✏️ Changed
~~~~~~~~~~

-  Changes some naming conventions for “amp_error” noise parameters to
   “calibration_error”

🐛 Fixed
~~~~~~~

-  Fixes several bugs with noise implementations in the simulator.
-  Fixes many spelling mistakes in simulator README.

.. _section-1:

`0.4.6`_ - 2018-01-22
---------------------

.. _changed-1:

✏️ Changed
~~~~~~~~~~

-  We have upgraded some of out external dependencies to:

   -  matplotlib >=2.1,<2.2
   -  networkx>=1.11,<2.1
   -  numpy>=1.13,<1.15
   -  ply==3.10
   -  scipy>=0.19,<1.1
   -  Sphinx>=1.6,<1.7
   -  sympy>=1.0

.. _section-2:

`0.4.4`_ - 2018-01-09
---------------------

.. _changed-2:

✏️ Changed
~~~~~~~~~~

-  Update dependencies to more recent versions

.. _fixed-1:

🐛 Fixed
~~~~~~~

-  Fix bug with process tomography reversing qubit preparation order

.. _section-3:

`0.4.3`_ - 2018-01-08
---------------------

⚠️ Deprecated
~~~~~~~~~~~~~

-  Static compilation has been removed because it seems to be failing
   while installing Qiskit via pip on Mac.

.. _section-4:

`0.4.2`_ - 2018-01-08
---------------------

.. _fixed-2:

🐛 Fixed
~~~~~~~

-  Minor bug fixing related to pip installation process.

.. _section-5:

`0.4.0`_ - 2018-01-08
---------------------

⚠️ TODO ⚠️

.. _Unreleased: https://github.com/QISKit/qiskit-core/compare/0.5.3...HEAD
.. _0.5.3: https://github.com/QISKit/qiskit-core/compare/0.5.2...0.5.3
.. _0.5.2: https://github.com/QISKit/qiskit-core/compare/0.5.1...0.5.2
.. _0.5.1: https://github.com/QISKit/qiskit-core/compare/0.5.0...0.5.1
.. _0.5.0: https://github.com/QISKit/qiskit-core/compare/0.4.15...0.5.0
.. _0.4.15: https://github.com/QISKit/qiskit-core/compare/0.4.14...0.4.15
.. _0.4.14: https://github.com/QISKit/qiskit-core/compare/0.4.13...0.4.14
.. _0.4.13: https://github.com/QISKit/qiskit-core/compare/0.4.12...0.4.13
.. _0.4.12: https://github.com/QISKit/qiskit-core/compare/0.4.11...0.4.12
.. _0.4.11: https://github.com/QISKit/qiskit-core/compare/0.4.10...0.4.11
.. _0.4.10: https://github.com/QISKit/qiskit-core/compare/0.4.9...0.4.10
.. _0.4.9: https://github.com/QISKit/qiskit-core/compare/0.4.8...0.4.9
.. _0.4.8: https://github.com/QISKit/qiskit-core/compare/0.4.7...0.4.8
.. _0.4.7: https://github.com/QISKit/qiskit-core/compare/0.4.6...0.4.7
.. _0.4.6: https://github.com/QISKit/qiskit-core/compare/0.4.5...0.4.6
.. _0.4.4: https://github.com/QISKit/qiskit-core/compare/0.4.3...0.4.4
.. _0.4.3: https://github.com/QISKit/qiskit-core/compare/0.4.2...0.4.3
.. _0.4.2: https://github.com/QISKit/qiskit-core/compare/0.4.1...0.4.2
.. _0.4.0: https://github.com/QISKit/qiskit-core/compare/0.3.16...0.4.0
