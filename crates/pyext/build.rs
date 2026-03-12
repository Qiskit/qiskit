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

use anyhow::anyhow;
use std::path::Path;

#[allow(clippy::print_stdout)] // We're a build script - we're _supposed_ to print to stdout.
fn main() -> anyhow::Result<()> {
    let cext_path = {
        let mut path = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        path.pop();
        path.push("cext");
        path
    };
    println!(
        "cargo::rerun-if-changed={}",
        cext_path
            .to_str()
            .ok_or_else(|| anyhow!("cext path isn't unicode"))?
    );
    let out_path = {
        let out_dir = std::env::var("OUT_DIR").expect("cargo should set this for build scripts");
        let mut path = Path::new(&out_dir).to_path_buf();
        path.push("include");
        path
    };
    let bindings = qiskit_bindgen::generate_bindings(&cext_path)?;
    // We install the headers into our `OUT_DIR`, then we configure `setuptools-rust` to pick them
    // up from there and put them into the Python package.
    qiskit_bindgen::install_c_headers(&bindings, &out_path)?;
    Ok(())
}
