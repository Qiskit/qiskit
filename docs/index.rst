###################################
Welcome to Qiskit's documentation!
###################################

Qiskit is an open-source framework for working with
noisy quantum computers
at the level of pulses, circuits, and algorithms.

A central goal of Qiskit is to build a software stack
that makes it easy for anyone to use quantum computers. However, Qiskit also aims
to facilitate research on the most important open issues facing quantum computation today.

You can use Qiskit to easily design experiments and run them on simulators and real
quantum computers.

Qiskit consists of four foundational elements:

- **Terra**: Composing quantum programs at the level of circuits and pulses with the
  code foundation
- **Aqua**: Building algorithms and applications
- **Ignis**: Addressing noise and errors
- **Aer**: Accelerating development via simulators, emulators, and debuggers

****************
Qiskit Elements
****************

.. image:: images/qiskit_elements.png


:ref:`Qiskit Terra <Terra>`
   Terra, the ‘earth’ element, is the foundation on which the rest of Qiskit lies.
   Terra provides a bedrock for composing quantum programs at the level of circuits and pulses,
   to optimize them for the constraints of a particular device, and to manage the execution
   of batches of experiments on remote-access devices. Terra defines the interfaces
   for a desirable end-user experience, as well as the efficient handling of layers
   of optimization, pulse scheduling and backend communication.


:ref:`Qiskit Aer <Aer>`
   Aer, the ‘air’ element, permeates all Qiskit elements. To really speed up development
   of quantum computers we need better simulators, emulators and debuggers.  Aer helps
   us understand the limits of classical processors by demonstrating to what extent they
   can mimic quantum computation. Furthermore, we can use Aer to verify that current
   and near-future quantum computers function correctly. This can be done by stretching
   the limits of simulation, and by simulating the effects of realistic noise on
   the computation.


:ref:`Qiskit Ignis <Ignis>`
   Ignis, the ‘fire’ element, is dedicated to fighting noise and errors and to forging
   a new path. This includes better characterization of errors, improving gates, and computing
   in the presence of noise. Ignis is meant for those who want to design quantum error
   correction codes, or who wish to study ways to characterize errors through methods
   such as tomography, or even to find a better way for using gates by exploring
   dynamical decoupling and optimal control.


:ref:`Qiskit Aqua <Aqua>`
   Aqua, the ‘water’ element, is the element of life. To make quantum computing live up
   to its expectations, we need to find real-world applications. Aqua is where algorithms
   for NISQ computers are built. These algorithms can be used to build applications
   for quantum computing. Aqua is accessible to domain experts in chemistry, optimization,
   finance and AI, who want to explore the benefits of using quantum computers as accelerators
   for specific computational tasks, without needing to worry about how to translate
   the problem into the language of quantum machines.

******************
Qiskit Components
******************

The components of Qiskit extend the functionality.

:ref:`Qiskit Chemistry <aqua-chemistry>`
   Qiskit Chemistry extends the Aqua element to allow the user to work easier
   with quantum computing for quantum chemistry problems. Qiskit Chemistry a set of tools,
   algorithms and software to use for quantum chemistry research. Qiskit Chemistry comes with
   interfaces prebuilt for the following four classical computational chemistry software drivers:

   - Gaussian™ 16 (a commercial chemistry program)
   - PSI4 (an open-source chemistry program built on Python)
   - PySCF (an open-source Python chemistry program)
   - PyQuante (a pure Python cross-platform open-source chemistry program).
:ref:`Qiskit AI <aqua-ai>`
   Qiskit Artificial Intelligence (AI) allows users with different levels of
   experience to execute AI experiments and contribute to the quantum computing AI
   software stack. Users with a pure AI background or interests can continue to
   configure AI problems without having to learn the details of quantum computing.
:ref:`Qiskit Optimization <aqua-optimization>`
   Qiskit Optimization allows users with different levels of experience to execute
   optimization experiments and contribute to the quantum computing optimization
   software stack. Users with a pure optimization background or interests can
   continue to configure optimization problems without having to learn the details
   of quantum computing.
:ref:`Qiskit Finance <aqua-finance>`
   Qiskit Finance allows users with different levels of experience to execute
   financial-analysis and optimization experiments and contribute to the quantum
   computing finance software stack. Users with a pure finance background or
   interests can continue to configure financial analysis and optimization problems
   without having to learn the details of the underlying quantum computing system.

.. toctree::
  :maxdepth: 2
  :hidden:

  Installing Qiskit <install>
  Getting Started <getting_started>
  Frequently Asked Questions <faq>
  Qiskit Terra <terra/index>
  Qiskit Aer <aer/index>
  Qiskit Ignis <ignis/index>
  Qiskit Aqua <aqua/index>
  Advanced Use of IBM Q Devices <advanced_use_of_ibm_q_devices>
  Development Strategy <development_strategy>
  Release Notes <release_notes>
  API References <autodoc/qiskit>
  Contributing to Qiskit <contributing_to_qiskit>
  Community Extensions <community>
  License <license>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`

