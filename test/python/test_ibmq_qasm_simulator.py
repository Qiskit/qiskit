    @requires_qe_access
    def test_execute_one_circuit_simulator_online(self, QE_TOKEN, QE_URL,
                                                  hub=None, group=None, project=None):
        """Test execute_one_circuit_simulator_online.

        If all correct should return the data.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("q", 1)
        cr = q_program.create_classical_register("c", 1)
        qc = q_program.create_circuit("qc", [qr], [cr])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        shots = 1024
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        backend = 'ibmq_qasm_simulator'
        result = q_program.execute(['qc'], backend=backend,
                                   shots=shots, max_credits=3,
                                   seed=73846087)
        counts = result.get_counts('qc')
        target = {'0': shots / 2, '1': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    @requires_qe_access
    def test_simulator_online_size(self, QE_TOKEN, QE_URL,
                                   hub=None, group=None, project=None):
        """Test test_simulator_online_size.

        If all correct should return the data.
        """
        backend_name = 'ibmq_qasm_simulator'
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("q", 31)
        cr = q_program.create_classical_register("c", 31)
        qc = q_program.create_circuit("qc", [qr], [cr])
        qc.h(qr)
        qc.measure(qr, cr)
        shots = 1
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        result = q_program.execute(['qc'], backend=backend_name, shots=shots,
                                   max_credits=3, seed=73846087)
        self.assertRaises(QISKitError, result.get_data, 'qc')
