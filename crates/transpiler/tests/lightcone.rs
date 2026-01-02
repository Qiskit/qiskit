#[cfg(test)]
mod tests {
    use pyo3::Python;
    use qiskit_transpiler::commutation_checker::CommutationChecker;

    #[test]
    fn smoke_test() {
        // Compilation succeeds = PauliEvolutionGate path compiles
        let _checker =
            qiskit_transpiler::commutation_checker::CommutationChecker::new(None, 1000, None);
    }
}
