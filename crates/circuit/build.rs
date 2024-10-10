// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use conan2::{ConanInstall, ConanVerbosity};
use std::{env, path::PathBuf};

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let conan_profile = format!("{}-{}", target_os, target_arch);
    
    let build_info = ConanInstall::new()
        .profile(&conan_profile)
        .detect_profile()
        .build("missing")
        .verbosity(ConanVerbosity::Error)
        .run()
        .parse();

    let mut include_paths = Vec::<String>::new();
    for path in build_info.include_paths() {
        let spath = path.into_os_string().into_string().unwrap();
        println!("{}", spath);
        if spath.contains("symen") {
            include_paths.push(spath);
        }
    }
    
    build_info.emit();

    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}/", include_paths[0]))
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings");
}