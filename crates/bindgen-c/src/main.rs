// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use clap::Parser;

/// Create a distribution of the "regular" C header files.
#[derive(Parser, Debug)]
struct Args {
    /// Path to the `cext` sources to generate headers for.
    cext_path: String,
    /// Where to install the header files to.
    install_path: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let bindings = qiskit_bindgen::generate_bindings(&args.cext_path)?;
    qiskit_bindgen::install_c_headers(&bindings, &args.install_path)?;
    Ok(())
}
