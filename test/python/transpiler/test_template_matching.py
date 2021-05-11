# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member


"""Test the TemplateOptimization pass."""

import unittest
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter, Gate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.circuit.library.templates import template_nct_2a_2, template_nct_5a_3
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.circuit_to_dagdependency import circuit_to_dagdependency
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import TemplateOptimization
from qiskit.test import QiskitTestCase
from qiskit.transpiler.exceptions import TranspilerError


class TestTemplateMatching(QiskitTestCase):
    """Test the TemplateOptimization pass."""

    def test_pass_cx_cancellation_no_template_given(self):
        """
        Check the cancellation of CX gates for the apply of the three basic
        template x-x, cx-cx. ccx-ccx.
        """
        qr = QuantumRegister(3)
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])

        pass_manager = PassManager()
        pass_manager.append(TemplateOptimization())
        circuit_in_opt = pass_manager.run(circuit_in)

        circuit_out = QuantumCircuit(qr)
        circuit_out.h(qr[0])
        circuit_out.h(qr[0])

        self.assertEqual(circuit_in_opt, circuit_out)

    def test_pass_cx_cancellation_own_template(self):
        """
        Check the cancellation of CX gates for the apply of a self made template cx-cx.
        """
        qr = QuantumRegister(2, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])
        dag_in = circuit_to_dag(circuit_in)

        qrt = QuantumRegister(2, "qrc")
        qct = QuantumCircuit(qrt)
        qct.cx(0, 1)
        qct.cx(0, 1)

        template_list = [qct]
        pass_ = TemplateOptimization(template_list)
        dag_opt = pass_.run(dag_in)

        circuit_expected = QuantumCircuit(qr)
        circuit_expected.h(qr[0])
        circuit_expected.h(qr[0])
        dag_expected = circuit_to_dag(circuit_expected)

        self.assertEqual(dag_opt, dag_expected)

    def test_pass_cx_cancellation_template_from_library(self):
        """
        Check the cancellation of CX gates for the apply of the library template cx-cx (2a_2).
        """
        qr = QuantumRegister(2, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])
        dag_in = circuit_to_dag(circuit_in)

        template_list = [template_nct_2a_2()]
        pass_ = TemplateOptimization(template_list)
        dag_opt = pass_.run(dag_in)

        circuit_expected = QuantumCircuit(qr)
        circuit_expected.h(qr[0])
        circuit_expected.h(qr[0])
        dag_expected = circuit_to_dag(circuit_expected)

        self.assertEqual(dag_opt, dag_expected)

    def test_pass_template_nct_5a(self):
        """
        Verify the result of template matching and substitution with the template 5a_3.
        q_0: ───────■─────────■────■──
                  ┌─┴─┐     ┌─┴─┐  │
        q_1: ──■──┤ X ├──■──┤ X ├──┼──
             ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
        q_2: ┤ X ├─────┤ X ├─────┤ X ├
             └───┘     └───┘     └───┘
        The circuit before optimization is:
              ┌───┐               ┌───┐
        qr_0: ┤ X ├───────────────┤ X ├─────
              └─┬─┘     ┌───┐┌───┐└─┬─┘
        qr_1: ──┼────■──┤ X ├┤ Z ├──┼────■──
                │    │  └─┬─┘└───┘  │    │
        qr_2: ──┼────┼────■────■────■────┼──
                │    │  ┌───┐┌─┴─┐  │    │
        qr_3: ──■────┼──┤ H ├┤ X ├──■────┼──
                │  ┌─┴─┐└───┘└───┘     ┌─┴─┐
        qr_4: ──■──┤ X ├───────────────┤ X ├
                   └───┘               └───┘

        The match is given by [0,1][1,2][2,7], after substitution the circuit becomes:

              ┌───┐               ┌───┐
        qr_0: ┤ X ├───────────────┤ X ├
              └─┬─┘     ┌───┐┌───┐└─┬─┘
        qr_1: ──┼───────┤ X ├┤ Z ├──┼──
                │       └─┬─┘└───┘  │
        qr_2: ──┼────■────■────■────■──
                │    │  ┌───┐┌─┴─┐  │
        qr_3: ──■────┼──┤ H ├┤ X ├──■──
                │  ┌─┴─┐└───┘└───┘
        qr_4: ──■──┤ X ├───────────────
                   └───┘
        """
        qr = QuantumRegister(5, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.ccx(qr[3], qr[4], qr[0])
        circuit_in.cx(qr[1], qr[4])
        circuit_in.cx(qr[2], qr[1])
        circuit_in.h(qr[3])
        circuit_in.z(qr[1])
        circuit_in.cx(qr[2], qr[3])
        circuit_in.ccx(qr[2], qr[3], qr[0])
        circuit_in.cx(qr[1], qr[4])
        dag_in = circuit_to_dag(circuit_in)

        template_list = [template_nct_5a_3()]
        pass_ = TemplateOptimization(template_list)
        dag_opt = pass_.run(dag_in)

        circuit_expected = QuantumCircuit(qr)
        circuit_expected.ccx(qr[3], qr[4], qr[0])
        circuit_expected.cx(qr[2], qr[4])
        circuit_expected.cx(qr[2], qr[1])
        circuit_expected.z(qr[1])
        circuit_expected.h(qr[3])
        circuit_expected.cx(qr[2], qr[3])
        circuit_expected.ccx(qr[2], qr[3], qr[0])

        dag_expected = circuit_to_dag(circuit_expected)

        self.assertEqual(dag_opt, dag_expected)

    def test_pass_template_wrong_type(self):
        """
        If a template is not equivalent to the identity, it raises an error.
        """
        qr = QuantumRegister(2, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])
        dag_in = circuit_to_dag(circuit_in)

        qrt = QuantumRegister(2, "qrc")
        qct = QuantumCircuit(qrt)
        qct.cx(0, 1)
        qct.x(0)
        qct.h(1)

        template_list = [qct]
        pass_ = TemplateOptimization(template_list)

        self.assertRaises(TranspilerError, pass_.run, dag_in)

    def test_accept_dagdependency(self):
        """
        Check that users can supply DAGDependency in the template list.
        """
        circuit_in = QuantumCircuit(2)
        circuit_in.cnot(0, 1)
        circuit_in.cnot(0, 1)

        templates = [circuit_to_dagdependency(circuit_in)]

        pass_ = TemplateOptimization(template_list=templates)
        circuit_out = PassManager(pass_).run(circuit_in)

        self.assertEqual(circuit_out.count_ops().get("cx", 0), 0)

    def test_parametric_template(self):
        """
        Check matching where template has parameters.
             ┌───────────┐                  ┌────────┐
        q_0: ┤ P(-1.0*β) ├──■────────────■──┤0       ├
             ├───────────┤┌─┴─┐┌──────┐┌─┴─┐│  CZ(β) │
        q_1: ┤ P(-1.0*β) ├┤ X ├┤ P(β) ├┤ X ├┤1       ├
             └───────────┘└───┘└──────┘└───┘└────────┘
        First test try match on
             ┌───────┐
        q_0: ┤ P(-2) ├──■────────────■─────────────────────────────
             ├───────┤┌─┴─┐┌──────┐┌─┴─┐┌───────┐
        q_1: ┤ P(-2) ├┤ X ├┤ P(2) ├┤ X ├┤ P(-3) ├──■────────────■──
             ├───────┤└───┘└──────┘└───┘└───────┘┌─┴─┐┌──────┐┌─┴─┐
        q_2: ┤ P(-3) ├───────────────────────────┤ X ├┤ P(3) ├┤ X ├
             └───────┘                           └───┘└──────┘└───┘
        Second test try match on
             ┌───────┐
        q_0: ┤ P(-2) ├──■────────────■────────────────────────────
             ├───────┤┌─┴─┐┌──────┐┌─┴─┐┌──────┐
        q_1: ┤ P(-2) ├┤ X ├┤ P(2) ├┤ X ├┤ P(3) ├──■────────────■──
             └┬──────┤└───┘└──────┘└───┘└──────┘┌─┴─┐┌──────┐┌─┴─┐
        q_2: ─┤ P(3) ├──────────────────────────┤ X ├┤ P(3) ├┤ X ├
              └──────┘                          └───┘└──────┘└───┘
        """

        class CZp(Gate):
            """CZ gates used for the test."""

            def __init__(self, num_qubits, params):
                super().__init__("cz", num_qubits, params)

            def inverse(self):
                inverse = UnitaryGate(np.diag([1.0, 1.0, 1.0, np.exp(-2.0j * self.params[0])]))
                inverse.name = "icz"
                return inverse

        def template_czp2():
            beta = Parameter("β")
            qc = QuantumCircuit(2)
            qc.p(-beta, 0)
            qc.p(-beta, 1)
            qc.cx(0, 1)
            qc.p(beta, 1)
            qc.cx(0, 1)
            qc.append(CZp(2, [beta]), [0, 1])

            return qc

        def count_cx(qc):
            """Counts the number of CX gates for testing."""
            return qc.count_ops().get("cx", 0)

        circuit_in = QuantumCircuit(3)
        circuit_in.p(-2, 0)
        circuit_in.p(-2, 1)
        circuit_in.cx(0, 1)
        circuit_in.p(2, 1)
        circuit_in.cx(0, 1)
        circuit_in.p(-3, 1)
        circuit_in.p(-3, 2)
        circuit_in.cx(1, 2)
        circuit_in.p(3, 2)
        circuit_in.cx(1, 2)

        pass_ = TemplateOptimization(template_list=[template_czp2()])
        circuit_out = PassManager(pass_).run(circuit_in)

        np.testing.assert_almost_equal(Operator(circuit_out).data[3, 3], np.exp(-4.0j))
        np.testing.assert_almost_equal(Operator(circuit_out).data[7, 7], np.exp(-10.0j))
        self.assertEqual(count_cx(circuit_out), 0)  # Two matches => no CX gates.
        np.testing.assert_almost_equal(Operator(circuit_in).data, Operator(circuit_out).data)

        circuit_in = QuantumCircuit(3)
        circuit_in.p(-2, 0)
        circuit_in.p(-2, 1)
        circuit_in.cx(0, 1)
        circuit_in.p(2, 1)
        circuit_in.cx(0, 1)
        circuit_in.p(3, 1)
        circuit_in.p(3, 2)
        circuit_in.cx(1, 2)
        circuit_in.p(3, 2)
        circuit_in.cx(1, 2)

        pass_ = TemplateOptimization(template_list=[template_czp2()])
        circuit_out = PassManager(pass_).run(circuit_in)

        self.assertEqual(count_cx(circuit_out), 2)  # One match => two CX gates.
        np.testing.assert_almost_equal(Operator(circuit_in).data, Operator(circuit_out).data)

    def test_unbound_parameters(self):
        """
        Test that partial matches with parameters will not raise errors.
        This tests that if parameters are still in the temporary template after
        _attempt_bind then they will not be used.
        """

        class PhaseSwap(Gate):
            """CZ gates used for the test."""

            def __init__(self, num_qubits, params):
                super().__init__("p", num_qubits, params)

            def inverse(self):
                inverse = UnitaryGate(
                    np.diag(
                        [1.0, 1.0, np.exp(-1.0j * self.params[0]), np.exp(-1.0j * self.params[0])]
                    )
                )
                inverse.name = "p"
                return inverse

        def template():
            beta = Parameter("β")
            qc = QuantumCircuit(2)
            qc.cx(1, 0)
            qc.cx(1, 0)
            qc.p(beta, 1)
            qc.append(PhaseSwap(2, [beta]), [0, 1])

            return qc

        circuit_in = QuantumCircuit(2)
        circuit_in.cx(1, 0)
        circuit_in.cx(1, 0)

        pass_ = TemplateOptimization(template_list=[template()])
        circuit_out = PassManager(pass_).run(circuit_in)

        # This template will not fully match as long as gates with parameters do not
        # commute with any other gates in the DAG dependency.
        self.assertEqual(circuit_out.count_ops().get("cx", 0), 2)


if __name__ == "__main__":
    unittest.main()
