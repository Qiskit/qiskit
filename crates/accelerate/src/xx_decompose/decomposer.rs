use std::f64::consts::PI;
use crate::xx_decompose::utilities::Square;
use crate::euler_one_qubit_decomposer::{self, EulerBasis};
use hashbrown::HashMap;

use super::types::Circuit2Q;

struct Point {
    a: f64,
    b: f64,
    c: f64,
}

/// Computes the infidelity distance between two points p, q expressed in positive canonical
/// coordinates.
fn _average_infidelity(p: Point, q: Point) -> f64 {
    let Point {
        a: a0,
        b: b0,
        c: c0,
    } = p;
    let Point {
        a: a1,
        b: b1,
        c: c1,
    } = q;

    1. - 1. / 20.
        * (4.
            + 16.
                * ((a0 - a1).cos().sq() * (b0 - b1).cos().sq() * (c0 - c1).cos().sq()
                    + (a0 - a1).sin().sq() * (b0 - b1).sin().sq() * (c0 - c1).sin().sq()))
}

// The Python class XXDecomposeer has an attribute `backup_optimizer`, which allows the
// caller to inject an alternative optimizer to run under certain conditions. For various
// reasons, I prefer omit this field. Instead, `XXDecomposer` can signal failure somehow.
pub(crate) struct XXDecomposer {
    basis_fidelity: Option<BasisFidelity>,
    euler_basis: EulerBasis,
    embodiments: Option<HashMap<f64, Circuit2Q>>,
}

// `f64` is not `Hash`. This is one way around.
struct BasisFidelity {
    data: HashMap<u64, f64>,
}

impl BasisFidelity {
    fn get(&self, k: f64) -> Option<f64> {
        self.data.get(&(k.to_bits())).copied()
    }
}

fn _strength_map_to_infidelity(strengths: &BasisFidelity, approximate: bool) -> BasisFidelity {
    let mapfunc = if approximate {
        |(strength, fidelity): (&u64, &f64) | (*strength, 1.0 - fidelity)
    } else {
        |(strength, fidelity): (&u64, &f64) | (*strength, 1e-12 + 1e-10 * f64::from_bits(*strength) / (PI / 2.0))
    };
    BasisFidelity { data:
                    strengths.data.iter()
                    .map(mapfunc)
                    .collect()
    }
}

// basis_fidelity: dict | float = 1.0,
// euler_basis: str = "U",
// embodiments: dict[float, QuantumCircuit] | None = None,
// backup_optimizer: Callable[..., QuantumCircuit] | None = None,
