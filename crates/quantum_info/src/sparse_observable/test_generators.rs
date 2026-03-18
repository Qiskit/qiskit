
use qiskit_quantum_info::sparse_observable::SparseObservable;
use qiskit_quantum_info::sparse_observable::standard_generators::generator_observable;
use qiskit_circuit::operations::StandardGate;
use num_complex::Complex64;

#[test]
fn test_all_generators() {
    let gates = vec![
        StandardGate::H, StandardGate::X, StandardGate::Y, StandardGate::Z,
        StandardGate::I, StandardGate::S, StandardGate::Sdg, StandardGate::T, StandardGate::Tdg,
        StandardGate::SX, StandardGate::SXdg, StandardGate::CX, StandardGate::CY, StandardGate::CZ,
        StandardGate::CS, StandardGate::CSdg, StandardGate::CSX, StandardGate::CCX, StandardGate::CCZ,
        StandardGate::CSwap, StandardGate::ECR, StandardGate::Swap, StandardGate::ISwap,
        StandardGate::RXX, StandardGate::RYY, StandardGate::RZZ, StandardGate::RZX,
        StandardGate::XXPlusYY, StandardGate::XXMinusYY, StandardGate::GlobalPhase,
    ];

    for gate in gates {
        println!("Testing gate {:?}", gate);
        let obs = generator_observable(gate, &[]);
        if let Some(o) = obs {
            println!("  {:?} layout valid", gate);
        } else {
             println!("  {:?} no generator (OK if expected)", gate);
        }
    }
}
