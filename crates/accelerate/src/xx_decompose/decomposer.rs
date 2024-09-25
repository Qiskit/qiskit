use crate::xx_decompose::utilities::Square;
use crate::euler_one_qubit_decomposer::EulerBasis;
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
    basis_fidelity: Option<HashMap<f64, f64>>,
    euler_basis: EulerBasis,
    embodiments: Option<HashMap<f64, Circuit2Q>>,
}

// basis_fidelity: dict | float = 1.0,
// euler_basis: str = "U",
// embodiments: dict[float, QuantumCircuit] | None = None,
// backup_optimizer: Callable[..., QuantumCircuit] | None = None,
