use crate::xx_decompose::utilities::Square;

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
