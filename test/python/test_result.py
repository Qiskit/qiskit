    def test_qubitpol(self):
        """Test the results of the qubitpol function in Results. Do two 2Q circuits
        in the first do nothing and in the second do X on the first qubit.
        """
        q_program = QuantumProgram()
        q = q_program.create_quantum_register("q", 2)
        c = q_program.create_classical_register("c", 2)
        qc1 = q_program.create_circuit("qc1", [q], [c])
        qc2 = q_program.create_circuit("qc2", [q], [c])
        qc2.x(q[0])
        qc1.measure(q, c)
        qc2.measure(q, c)
        circuits = ['qc1', 'qc2']
        xvals_dict = {circuits[0]: 0, circuits[1]: 1}

        result = q_program.execute(circuits, backend='local_qasm_simulator')

        yvals, xvals = result.get_qubitpol_vs_xval(2, xvals_dict=xvals_dict)

        self.assertTrue(np.array_equal(yvals, [[-1, -1], [1, -1]]))
        self.assertTrue(np.array_equal(xvals, [0, 1]))


    def test_average_data(self):
        """Test average_data.

        If all correct should return the data.
        """
        q_program = QuantumProgram()
        q = q_program.create_quantum_register("q", 2)
        c = q_program.create_classical_register("c", 2)
        qc = q_program.create_circuit("qc", [q], [c])
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        circuits = ['qc']
        shots = 10000
        backend = 'local_qasm_simulator'
        results = q_program.execute(circuits, backend=backend, shots=shots)
        observable = {"00": 1, "11": 1, "01": -1, "10": -1}
        mean_zz = results.average_data("qc", observable)
        observable = {"00": 1, "11": -1, "01": 1, "10": -1}
        mean_zi = results.average_data("qc", observable)
        observable = {"00": 1, "11": -1, "01": -1, "10": 1}
        mean_iz = results.average_data("qc", observable)
        self.assertAlmostEqual(mean_zz, 1, places=1)
        self.assertAlmostEqual(mean_zi, 0, places=1)
        self.assertAlmostEqual(mean_iz, 0, places=1)

    def test_combine_results(self):
        """Test run.

        If all correct should the data.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("qr", 1)
        cr = q_program.create_classical_register("cr", 1)
        qc1 = q_program.create_circuit("qc1", [qr], [cr])
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc1.measure(qr[0], cr[0])
        qc2.x(qr[0])
        qc2.measure(qr[0], cr[0])
        shots = 1024
        backend = 'local_qasm_simulator'
        res1 = q_program.execute(['qc1'], backend=backend, shots=shots)
        res2 = q_program.execute(['qc2'], backend=backend, shots=shots)
        counts1 = res1.get_counts('qc1')
        counts2 = res2.get_counts('qc2')
        res1 += res2  # combine results
        counts12 = [res1.get_counts('qc1'), res1.get_counts('qc2')]
        self.assertEqual(counts12, [counts1, counts2])


