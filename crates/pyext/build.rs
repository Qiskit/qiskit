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

use cbindgen::bindgen::ir;
use hashbrown::HashMap;
use qiskit_cext_vtable::{FUNCTIONS_CIRCUIT, FUNCTIONS_QI, FUNCTIONS_TRANSPILE};
use std::fs;
use std::io::Write;
use std::path::Path;

static WRAPPER_FUNCS: &str = "funcs_py.h";
static GENERATED_FUNCS: &str = "funcs_py_generated.h";

/// Render a given type object into a string representing it in C.
fn render_type_as_c(ty: &ir::Type, config: &cbindgen::Config) -> String {
    fn render(ty: &ir::Type, config: &cbindgen::Config, acc: &mut String) {
        dbg!(ty);
        match ty {
            ir::Type::Ptr {
                ty,
                is_const,
                is_nullable: _,
                is_ref,
            } => {
                assert!(!is_ref, "C++ reference-likes not handled");
                if *is_const {
                    acc.push_str("const ");
                }
                render(ty, config, acc);
                acc.push_str(" *");
            }
            ir::Type::Path(p) => acc.push_str(p.export_name()),
            ir::Type::Primitive(ty) => acc.push_str(ty.to_repr_c(config)),
            ir::Type::Array(..) => todo!("array types not yet handled"),
            ir::Type::FuncPtr {
                args,
                ret,
                is_nullable,
                never_return,
            } => {
                assert!(!is_nullable, "nullability of funcptrs is not handled");
                assert!(!never_return, "diverging functions not handled");
                render(ret, config, acc);
                acc.push_str("(*)(");
                let mut args = args.iter();
                if let Some((_, first)) = args.next() {
                    render(first, config, acc);
                    for (_, arg) in args {
                        acc.push_str(", ");
                        render(arg, config, acc)
                    }
                }
                acc.push(')');
            }
        }
    }
    let mut acc = String::new();
    render(ty, config, &mut acc);
    acc
}

/// Calculate a mapping of exported function names to C casts to appropriate function-pointer types.
fn functions_as_c_funcptr_casts(bindings: &cbindgen::Bindings) -> HashMap<&str, String> {
    let to_funcptr = |func: &ir::Function| {
        let to_funcptr_arg = |arg: &ir::FunctionArgument| {
            let ir::FunctionArgument {
                name: _,
                ty,
                array_length,
            } = arg;
            assert!(array_length.is_none(), "array arguments not handled");
            (None, ty.clone())
        };
        ir::Type::FuncPtr {
            ret: Box::new(func.ret.clone()),
            args: func.args.iter().map(to_funcptr_arg).collect(),
            is_nullable: false,
            never_return: false,
        }
    };
    let config = &bindings.config;
    bindings
        .functions
        .iter()
        .map(|func| {
            let funcptr = to_funcptr(func);
            (func.path.name(), render_type_as_c(&funcptr, config))
        })
        .collect()
}

/// Install (overwriting) the Python-extension-specific header files into the given directory.
fn install_py_function_headers(
    bindings: &cbindgen::Bindings,
    install_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let mut our_include = Path::new(env!("CARGO_MANIFEST_DIR")).join("include");
    our_include.push(qiskit_bindgen::SCOPED_INCLUDE_DIR);
    // This directory must already have been constructed by the previous "install" command for the
    // regular C headers; if it doesn't, writing out the files will be a mistake because we _should_
    // be overwriting an existing file (the wrapper that defines `qk_import`).
    let install_path = install_path
        .as_ref()
        .join(qiskit_bindgen::SCOPED_INCLUDE_DIR);
    fs::copy(
        our_include.join(WRAPPER_FUNCS),
        install_path.join(WRAPPER_FUNCS),
    )?;
    let mut funcs_header = fs::File::create(install_path.join(GENERATED_FUNCS))?;
    writeln!(funcs_header, "{}", qiskit_bindgen::COPYRIGHT)?;

    // Now, each function's name is just a preprocessor macro that resolves to a lookup into the
    // corresponding table.  The names given here need to match with the handwritten include file
    // that sets up the slots in `qk_import`.
    let vtables = [
        ("_Qk_API_Circuit", &FUNCTIONS_CIRCUIT),
        ("_Qk_API_Transpile", &FUNCTIONS_TRANSPILE),
        ("_Qk_API_QI", &FUNCTIONS_QI),
    ];
    let funcs = functions_as_c_funcptr_casts(bindings);
    for (vtable_name, vtable) in vtables {
        for export in vtable.exports(0) {
            writeln!(
                funcs_header,
                "#define {} (*({})({}[{}]))",
                export.name, funcs[export.name], vtable_name, export.slot
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::print_stdout)] // We're a build script - we're _supposed_ to print to stdout.
fn main() -> anyhow::Result<()> {
    // Our actual requirements for re-running the build script are if `cext-vtable` changes, but
    // since that's a build-time dependency, it's already implicit in Cargo's logic, so we just need
    // to issue _any_ re-run command to avoid the default behaviour of rerunning if `qiskit_pyext`
    // itself changes.
    println!("cargo::rerun-if-changed=build.rs");
    let cext_path = {
        let mut path = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        path.pop();
        path.push("cext");
        path
    };
    let out_path = {
        let out_dir = std::env::var("OUT_DIR").expect("cargo should set this for build scripts");
        let mut path = Path::new(&out_dir).to_path_buf();
        path.push("include");
        path
    };
    let mut bindings = qiskit_bindgen::generate_bindings(&cext_path)?;
    // We install the headers into our `OUT_DIR`, then we configure `setuptools-rust` to pick them
    // up from there and put them into the Python package.
    qiskit_bindgen::install_c_headers(&mut bindings, &out_path)?;
    install_py_function_headers(&bindings, &out_path)?;
    Ok(())
}
