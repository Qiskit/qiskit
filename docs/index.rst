.. raw:: html

  <div style="width: 100%; align: center">

.. raw:: html
  :file: images/qiskit_main.svg

.. raw:: html

  </div>

##############################
Qiskit |version| documentation
##############################

Qiskit is open-source software for working with quantum computers
at the level of circuits, pulses, and algorithms.  Additionally, several
domain specific application API's exist on top of this core module.

The central goal of Qiskit is to build a software stack
that makes it easy for anyone to use quantum computers, regardless of their skill level or
area of interest; Qiskit allows one to easily design experiments and applications and run
them on real quantum computers and/or classical simulators.  Qiskit is already in use
around the world by beginners, hobbyists, educators, researchers, and commercial companies.

.. raw:: html

  <div class="tutorials-callout-container">
     <div class="row">

.. customcalloutitem::
  :description: Find out which Qiskit Partners support execution on real quantum services.
  :header: Access to quantum systems
  :button_link:  https://qiskit.org/documentation/partners/
  :button_text: Qiskit Partners


.. customcalloutitem::
  :description: A programming model and execution framework to effectively execute workloads.
  :header: Qiskit Runtime
  :button_link:  https://qiskit.org/documentation/partners/qiskit_ibm_runtime/
  :button_text: Get started

.. raw:: html

  </div>

Interested in applications of quantum computing?
################################################

.. raw:: html

  <div class="applications-callout-container">
     <div class="row">

.. customcalloutitem::
  :description: Qiskit Nature supports different applications, such as electronic/vibronic structure calculations for ground and excited states or protein folding. It provides all the components necessary to interface classical codes and automatically convert to different representations required by quantum computers.
  :header: Qiskit Nature
  :button_link: https://qiskit.org/documentation/nature/
  :button_text: Qiskit Nature

.. customcalloutitem::
  :description: Qiskit Finance provides a set of illustrative applications and tools, including Ising translators for portfolio optimization, data providers for real or random data, and implementations for pricing different financial options or for credit risk analysis.
  :header: Qiskit Finance
  :button_link: https://qiskit.org/documentation/finance/
  :button_text: Qiskit Finance

.. customcalloutitem::
  :description: Qiskit Machine Learning provides fundamental quantum kernels and quantum neural networks (QNNs) as building blocks and quantum machine learning algorithms that apply them to solve different tasks like regression and classification. Further, it allows to connect QNNs to PyTorch to enhance classical workflows with quantum components.
  :header: Qiskit Machine Learning
  :button_link: https://qiskit.org/documentation/machine-learning/
  :button_text: Qiskit Machine Learning

.. customcalloutitem::
  :description: Qiskit Optimization provides the whole range from high-level modeling of optimization problems, to automatic conversion of problems to different required representations, to a suite of easy-to-use quantum optimization algorithms.
  :header: Qiskit Optimization
  :button_link: https://qiskit.org/documentation/optimization/
  :button_text: Qiskit Optimization

.. raw:: html

  </div>

Interested in running experiments on real qubits?
#################################################

.. customcalloutitem::
  :description: Run characterization, calibration, and verification experiments
  :header: Qiskit Experiments
  :button_link: https://qiskit.org/documentation/experiments/
  :button_text: Qiskit Experiments

Interested in quantum hardware design?
######################################

   .. grid:: 2

    .. grid-item-card::
        :columns: auto

        Qiskit Metal
        ^^^^^^^^^^^^^^

        .. image:: images/metal.png
          :scale: 35 %
          :align: center
          :target: https://qiskit.org/documentation/metal/

        ++++++
        :bdg-link-primary-line:`Website <https://qiskit.org/metal>`
        :bdg-link-primary-line:`Documentation <https://qiskit.org/documentation/metal/>`

.. toctree::
   :hidden:

   Documentation Home <self>
   qc_intro
   getting_started
   intro_tutorial1
   tutorials
   API Reference <apidoc/terra>
   Circuit Library <apidoc/circuit_library>
   release_notes
   configuration
   GitHub <https://github.com/Qiskit/qiskit-terra>
   faq

.. toctree::
   :caption: Contributing
   :hidden:

   contributing_to_qiskit
   deprecation_policy
   maintainers_guide

.. toctree::
   :caption: Other API References
   :hidden:

   Qiskit Aer <apidoc/aer>
   Qiskit IBM Quantum Provider (deprecated) <apidoc/ibmq-provider>
