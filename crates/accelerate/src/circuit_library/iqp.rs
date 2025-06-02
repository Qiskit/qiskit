pub fn iqp_circuit(num_qubits: usize, reps: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(num_qubits);
    
    // Generate random symmetric interaction matrix for each repetition
    let mut rng = Pcg64Mcg::from_entropy();
    let mut interactions = Vec::with_capacity(reps);
    
    for _ in 0..reps {
        let mut mat = Array2::zeros((num_qubits, num_qubits));
        for i in 0..num_qubits {
            mat[[i, i]] = rng.gen_range(0..8);
            for j in i+1..num_qubits {
                let val = rng.gen_range(0..8);
                mat[[i, j]] = val;
                mat[[j, i]] = val;
            }
        }
        interactions.push(mat);
    }

    // Initial Hadamard layer
    for qubit in 0..num_qubits {
        circuit.h(qubit);
    }

    // Add repeated interaction layers
    for mat in interactions {
        // Phase shifts (diagonal elements)
        for i in 0..num_qubits {
            circuit.append(
                StandardGate::PhaseGate,
                smallvec![Param::Float(PI4 * mat[[i, i]] as f64)],
                smallvec![Qubit(i as u32)]
            );
        }

        // Entanglement layer (upper triangular elements)
        for i in 0..num_qubits {
            for j in i+1..num_qubits {
                if mat[[i, j]] % 4 != 0 {
                    circuit.append(
                        StandardGate::CPhaseGate,
                        smallvec![Param::Float(PI2 * mat[[i, j]] as f64)],
                        smallvec![Qubit(i as u32), Qubit(j as u32)]
                    );
                }
            }
        }
    }

    // Final Hadamard layer
    for qubit in 0..num_qubits {
        circuit.h(qubit);
    }

    circuit
}
