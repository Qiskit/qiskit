    def verify_unitary():
        # Test gate sequence
        V = np.identity(4, dtype=complex)
        cx21 = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]], dtype=complex)

        cx12 = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)

        for gate in return_circuit:
            if gate["name"] == "cx":
                if gate["args"] == [0, 1]:
                    V = np.dot(cx12, V)
                else:
                    V = np.dot(cx21, V)
            else:
                if gate["args"] == [0]:
                    V = np.dot(np.kron(rz_array(gate["params"][2]),
                                       np.identity(2)), V)
                    V = np.dot(np.kron(ry_array(gate["params"][0]),
                                       np.identity(2)), V)
                    V = np.dot(np.kron(rz_array(gate["params"][1]),
                                       np.identity(2)), V)
                else:
                    V = np.dot(np.kron(np.identity(2),
                                       rz_array(gate["params"][2])), V)
                    V = np.dot(np.kron(np.identity(2),
                                       ry_array(gate["params"][0])), V)
                    V = np.dot(np.kron(np.identity(2),
                                       rz_array(gate["params"][1])), V)
        # Put V in SU(4) and test up to global phase
        V = la.det(V)**(-1.0/4.0) * V
        if la.norm(V - U) > 1e-6 and \
           la.norm(1j*V - U) > 1e-6 and \
           la.norm(-1*V - U) > 1e-6 and \
           la.norm(-1j*V - U) > 1e-6:
            raise QiskitError("two_qubit_kak: Circuit implementation" +
                              "does not match input unitary.")
