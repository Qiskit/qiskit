use num_complex::Complex64;
use qiskit_circuit::operations::{StandardGate, STANDARD_GATE_SIZE};

use super::BitTerm;
use super::SparseObservable;

/// Return an observable for the generator of `gate`, if we have one.
///
/// `None` means “no special handling, use the generic commutation path”.
// Define constant slices for the terms to be used in the lookup table
const TERMS_X: &[BitTerm] = &[BitTerm::X];
const TERMS_Y: &[BitTerm] = &[BitTerm::Y];
const TERMS_Z: &[BitTerm] = &[BitTerm::Z];
const TERMS_XZ: &[BitTerm] = &[BitTerm::X, BitTerm::Z];

const fn build_generator_lut() -> [Option<&'static [BitTerm]>; STANDARD_GATE_SIZE] {
    let mut lut = [None; STANDARD_GATE_SIZE];
    
    // X-like
    lut[StandardGate::X as usize] = Some(TERMS_X);
    lut[StandardGate::SX as usize] = Some(TERMS_X);
    lut[StandardGate::SXdg as usize] = Some(TERMS_X);
    lut[StandardGate::RX as usize] = Some(TERMS_X);

    // Y-like
    lut[StandardGate::Y as usize] = Some(TERMS_Y);
    lut[StandardGate::RY as usize] = Some(TERMS_Y);

    // Z-like
    lut[StandardGate::Z as usize] = Some(TERMS_Z);
    lut[StandardGate::S as usize] = Some(TERMS_Z);
    lut[StandardGate::Sdg as usize] = Some(TERMS_Z);
    lut[StandardGate::T as usize] = Some(TERMS_Z);
    lut[StandardGate::Tdg as usize] = Some(TERMS_Z);
    lut[StandardGate::RZ as usize] = Some(TERMS_Z);
    lut[StandardGate::Phase as usize] = Some(TERMS_Z);
    lut[StandardGate::U1 as usize] = Some(TERMS_Z);

    // H
    lut[StandardGate::H as usize] = Some(TERMS_XZ);
    
    lut
}

static GENERATOR_LUT: [Option<&'static [BitTerm]>; STANDARD_GATE_SIZE] = build_generator_lut();

/// Return an observable for the generator of `gate`, if we have one.
///
/// `None` means “no special handling, use the generic commutation path”.
pub fn generator_observable(gate: StandardGate) -> Option<SparseObservable> {
    let terms: &[BitTerm] = GENERATOR_LUT[gate as usize]?;

    // For now assume a single-qubit generator acting on 1 qubit.
    // TODO: Once we're satisfied with the pipeline (LightCone pass works for large numbers of qubits
    // and ASV benchmarks show we are not slower), we should add all other standard gates here.
    let num_qubits = 1;

    // One coefficient per term, all +1 for now.
    let coeffs = vec![Complex64::new(1.0, 0.0); terms.len()];

    // Flatten the BitTerm slice into a Vec<BitTerm>.
    let bit_terms: Vec<BitTerm> = terms.to_vec();

    // Each term uses one BitTerm; for a 1-qubit gate, all terms act on qubit 0.
    let indices: Vec<u32> = vec![0; bit_terms.len()];
    let boundaries: Vec<usize> = (0..=bit_terms.len()).collect();

    // Safe constructor; unwrap is fine here because we control the layout.
    let obs = SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)
        .expect("invalid generator observable layout");

    Some(obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rx_has_some_generator() {
        let obs = generator_observable(StandardGate::RX).expect("RX should have a generator");
        assert!(!obs.bit_terms().is_empty());
    }
}
