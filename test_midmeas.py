if __name__ == "__main__":
    from math import pi
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RXGate, XGate, UGate
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
            ASAPScheduleAnalysis,
            ALAPScheduleAnalysis,
            PadDynamicalDecoupling,
        )
    from qiskit.transpiler.passes.scheduling.padding.dynamical_decoupling import (
        PadDynamicalDecoupling,
    )
    from qiskit.transpiler.instruction_durations import InstructionDurations
    import traceback

    midmeas = QuantumCircuit(3, 1)
    midmeas.cx(0, 1)
    midmeas.cx(1, 2)
    midmeas.u(pi, 0, pi, 0)
    midmeas.measure(2, 0)
    midmeas.cx(1, 2)
    midmeas.cx(0, 1)

    durations = InstructionDurations(
        [
            ("h", 0, 50),
            ("cx", [0, 1], 700),
            ("cx", [1, 2], 200),
            ("cx", [2, 3], 300),
            ("x", None, 50),
            ("y", None, 50),
            ("u", None, 100),
            ("rx", None, 100),
            ("measure", None, 1000),
            ("reset", None, 1500),
        ],
        dt=1e-9,
    )
    try:
        """Test a single X gate as Hahn echo can absorb in the downstream circuit.

        global phase: 3π/2
                               ┌────────────────┐       ┌───┐       ┌────────────────┐»
        q_0: ────────■─────────┤ Delay(625[dt]) ├───────┤ X ├───────┤ Delay(625[dt]) ├»
                   ┌─┴─┐       └────────────────┘┌──────┴───┴──────┐└────────────────┘»
        q_1: ──────┤ X ├───────────────■─────────┤ Delay(1000[dt]) ├────────■─────────»
             ┌─────┴───┴──────┐      ┌─┴─┐       └───────┬─┬───────┘      ┌─┴─┐       »
        q_2: ┤ Delay(700[dt]) ├──────┤ X ├───────────────┤M├──────────────┤ X ├───────»
             └────────────────┘      └───┘               └╥┘              └───┘       »
        c: 1/═════════════════════════════════════════════╩═══════════════════════════»
                                                          0                           »
        «     ┌───────────────┐
        «q_0: ┤ U(0,π/2,-π/2) ├───■──
        «     └───────────────┘ ┌─┴─┐
        «q_1: ──────────────────┤ X ├
        «     ┌────────────────┐└───┘
        «q_2: ┤ Delay(700[dt]) ├─────
        «     └────────────────┘
        «c: 1/═══════════════════════
        """
        dd_sequence = [XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(durations, dd_sequence),
            ]
        )

        midmeas_dd = pm.run(midmeas)

        combined_u = UGate(0, 0, 0)

        expected = QuantumCircuit(3, 1)
        expected.cx(0, 1)
        expected.delay(625, 0)
        expected.x(0)
        expected.delay(625, 0)
        expected.compose(combined_u, [0], inplace=True)
        expected.delay(700, 2)
        expected.cx(1, 2)
        expected.delay(1000, 1)
        expected.measure(2, 0)
        expected.cx(1, 2)
        expected.cx(0, 1)
        expected.delay(700, 2)
        expected.global_phase = pi

        midmeas_dd == expected
        print("computed circuit", midmeas_dd.draw(output="text"))
        print("expected circuit", expected.draw(output="text"))

        # check the absorption into U was done correctly
        Operator(combined_u) == Operator(XGate()) & Operator(XGate())
    except Exception as e:
        print("An error occurred during the test:")
        traceback.print_exc()