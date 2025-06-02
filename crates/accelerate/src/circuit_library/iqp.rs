pub fn iqp_circuit(num_qubits: usize, reps: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(num_qubits);
    
    // Apply Hadamard layer
    for qubit in 0..num_qubits {
        circuit.h(qubit);
    }

    // Generate entanglement pairs (unchanged)
    let mut pairs = Vec::new();
    for i in 0..num_qubits {
        for j in i+1..num_qubits {
            pairs.push((i, j));
        }
    }

    // Add ZZGate layers with per-repetition parameters
    for _ in 0..reps {
        // Generate new parameters for each rep
        let mut params = Vec::new();
        for i in 0..num_qubits {
            for j in i+1..num_qubits {
                params.push((i * num_qubits + j) as f64 / 2.0);
            }
        }
        
        circuit.append(
            NLocal::new(
                vec![Gate::ZZ], 
                params,  // Use local params, not cloned
                pairs.clone()
            )
        );
    }

    circuit
}
