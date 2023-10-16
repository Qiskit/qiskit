/// Return indices that sort data.
/// If `data` contains two elements that are incomparable,
/// an error will be thrown.
pub fn argsort<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

// TODO: Use traits and parameters
/// Modulo operation
pub fn modulo(a: f64, b: f64) -> f64 {
    return ((a % b) + b) % b;
}
