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


"""Test the TemplateOptimization pass."""

import unittest
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.circuit.library.templates import template_nct_2a_2, template_nct_5a_3
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.circuit_to_dagdependency import circuit_to_dagdependency
from qiskit.qasm import pi
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import TemplateOptimization
from qiskit.transpiler.passes.calibration.rzx_templates import rzx_templates
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
             ├───────────┤┌─┴─┐┌──────┐┌─┴─┐│  CU(2β)│
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

        beta = Parameter("β")
        template = QuantumCircuit(2)
        template.p(-beta, 0)
        template.p(-beta, 1)
        template.cx(0, 1)
        template.p(beta, 1)
        template.cx(0, 1)
        template.cu(0, 2.0 * beta, 0, 0, 0, 1)

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

        pass_ = TemplateOptimization(
            template_list=[template],
            user_cost_dict={"cx": 6, "p": 0, "cu": 8},
        )
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

        pass_ = TemplateOptimization(
            template_list=[template],
            user_cost_dict={"cx": 6, "p": 0, "cu": 8},
        )
        circuit_out = PassManager(pass_).run(circuit_in)

        self.assertEqual(count_cx(circuit_out), 2)  # One match => two CX gates.
        np.testing.assert_almost_equal(Operator(circuit_in).data, Operator(circuit_out).data)

    def test_optimizer_does_not_replace_unbound_partial_match(self):
        """
        Test that partial matches with parameters will not raise errors.
        This tests that if parameters are still in the temporary template after
        _attempt_bind then they will not be used.
        """

        beta = Parameter("β")
        template = QuantumCircuit(2)
        template.cx(1, 0)
        template.cx(1, 0)
        template.p(beta, 1)
        template.cu(0, 0, 0, -beta, 0, 1)

        circuit_in = QuantumCircuit(2)
        circuit_in.cx(1, 0)
        circuit_in.cx(1, 0)
        pass_ = TemplateOptimization(
            template_list=[template],
            user_cost_dict={"cx": 6, "p": 0, "cu": 8},
        )

        circuit_out = PassManager(pass_).run(circuit_in)

        # The template optimisation should not have replaced anything, because
        # that would require it to leave dummy parameters in place without
        # binding them.
        self.assertEqual(circuit_in, circuit_out)

    def test_unbound_parameters_in_rzx_template(self):
        """
        Test that rzx template ('zz2') functions correctly for a simple
        circuit with an unbound ParameterExpression. This uses the same
        Parameter (theta) as the template, so this also checks that template
        substitution handle this correctly.
        """

        theta = Parameter("ϴ")
        circuit_in = QuantumCircuit(2)
        circuit_in.cx(0, 1)
        circuit_in.p(2 * theta, 1)
        circuit_in.cx(0, 1)

        pass_ = TemplateOptimization(**rzx_templates(["zz2"]))
        circuit_out = PassManager(pass_).run(circuit_in)

        # these are NOT equal if template optimization works
        self.assertNotEqual(circuit_in, circuit_out)

        # however these are equivalent if the operators are the same
        theta_set = 0.42
        self.assertTrue(
            Operator(circuit_in.bind_parameters({theta: theta_set})).equiv(
                circuit_out.bind_parameters({theta: theta_set})
            )
        )

    def test_two_parameter_template(self):
        """
        Test a two-Parameter template based on rzx_templates(["zz3"]),

                                ┌───┐┌───────┐┌───┐┌────────────┐»
        q_0: ──■─────────────■──┤ X ├┤ Rz(φ) ├┤ X ├┤ Rz(-1.0*φ) ├»
             ┌─┴─┐┌───────┐┌─┴─┐└─┬─┘└───────┘└─┬─┘└────────────┘»
        q_1: ┤ X ├┤ Rz(θ) ├┤ X ├──■─────────────■────────────────»
             └───┘└───────┘└───┘
        «     ┌─────────┐┌─────────┐┌─────────┐┌───────────┐┌──────────────┐»
        «q_0: ┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├┤ Rx(1.0*φ) ├┤1             ├»
        «     └─────────┘└─────────┘└─────────┘└───────────┘│  Rzx(-1.0*φ) │»
        «q_1: ──────────────────────────────────────────────┤0             ├»
        «                                                   └──────────────┘»
        «      ┌─────────┐  ┌─────────┐┌─────────┐                        »
        «q_0: ─┤ Rz(π/2) ├──┤ Rx(π/2) ├┤ Rz(π/2) ├────────────────────────»
        «     ┌┴─────────┴─┐├─────────┤├─────────┤┌─────────┐┌───────────┐»
        «q_1: ┤ Rz(-1.0*θ) ├┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├┤ Rx(1.0*θ) ├»
        «     └────────────┘└─────────┘└─────────┘└─────────┘└───────────┘»
        «     ┌──────────────┐
        «q_0: ┤0             ├─────────────────────────────────
        «     │  Rzx(-1.0*θ) │┌─────────┐┌─────────┐┌─────────┐
        «q_1: ┤1             ├┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├
        «     └──────────────┘└─────────┘└─────────┘└─────────┘

        correctly template matches into a unique circuit, but that it is
        equivalent to the input circuit when the Parameters are bound to floats
        and checked with Operator equivalence.
        """
        theta = Parameter("θ")
        phi = Parameter("φ")

        template = QuantumCircuit(2)
        template.cx(0, 1)
        template.rz(theta, 1)
        template.cx(0, 1)
        template.cx(1, 0)
        template.rz(phi, 0)
        template.cx(1, 0)
        template.rz(-phi, 0)
        template.rz(pi / 2, 0)
        template.rx(pi / 2, 0)
        template.rz(pi / 2, 0)
        template.rx(phi, 0)
        template.rzx(-phi, 1, 0)
        template.rz(pi / 2, 0)
        template.rz(-theta, 1)
        template.rx(pi / 2, 0)
        template.rz(pi / 2, 1)
        template.rz(pi / 2, 0)
        template.rx(pi / 2, 1)
        template.rz(pi / 2, 1)
        template.rx(theta, 1)
        template.rzx(-theta, 0, 1)
        template.rz(pi / 2, 1)
        template.rx(pi / 2, 1)
        template.rz(pi / 2, 1)

        alpha = Parameter("$\\alpha$")
        beta = Parameter("$\\beta$")

        circuit_in = QuantumCircuit(2)
        circuit_in.cx(0, 1)
        circuit_in.rz(2 * alpha, 1)
        circuit_in.cx(0, 1)
        circuit_in.cx(1, 0)
        circuit_in.rz(3 * beta, 0)
        circuit_in.cx(1, 0)

        pass_ = TemplateOptimization(
            [template],
            user_cost_dict={"cx": 6, "rz": 0, "rx": 1, "rzx": 0},
        )
        circuit_out = PassManager(pass_).run(circuit_in)

        # these are NOT equal if template optimization works
        self.assertNotEqual(circuit_in, circuit_out)

        # however these are equivalent if the operators are the same
        alpha_set = 0.39
        beta_set = 0.42
        self.assertTrue(
            Operator(circuit_in.bind_parameters({alpha: alpha_set, beta: beta_set})).equiv(
                circuit_out.bind_parameters({alpha: alpha_set, beta: beta_set})
            )
        )


if __name__ == "__main__":
    unittest.main()
