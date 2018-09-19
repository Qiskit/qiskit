## Goals
The main goal is to provide an extensible infrastructure of pluggable passes that allows flexibility in customizing the translation pipeline through the creation and combination of new passes.

## General overview

- The transpiler is pluggable translation pipeline.
- The `PassManager` class represents the pipeline.

### Passes
- Passes are instances of either `AnalysisPass` or `TransformationPass`.
- Passes run with the implementation of the abstract method `run`, which take a DAG representation of the circuit as a parameter.
- Analysis passes analyze the DAG and write conclusions to a common context, an instance of the `PropertySet` class.
- All the passes can read the property set.
- Transformation passes can alter the DAG and should return a DAG.

### Pass Mananger
- A `PassManager` instance schedule for running registered passes.
- A PassManager is in charge of deciding which is the next pass, not the pass itself.
- The registration is performed by `add_pass` method.
- While registering, you can specify basic control primitives to the passes (conditional and loop passes).
- Options to control the scheduler
	- The precedence of passes options is pass, pass set, pass manager. (see [tests](https://github.com/Qiskit/qiskit-terra/compare/master...1ucian0:transpiler?expand=1#diff-086bfc6396298d112141bcb72d7d76ddR236))
	- Passes can have arguments at init-time that can be used during run-time. That's why, if you want to set properties related on how the pass is run (for example, if it's idempotent), the method `pass_.set()` should be used).



### Pass dependency control
Regarding passes dependencies, the architecture features two kinds of dependencies:

- `requires` are passes that need to have been run before executing the current pass.
- `preserves` are passes that are not invalidated by the current pass.
- Analysis passes preserve all.
- The `preserves` and `requires` lists are passes with arguments.
- Passes are described not just by their name, but also by their parameters (see Use cases, pass identity)



## Use cases
### A simple chain with dependencies:
The `CxCancellation` requires and preserves `ToffoliDecompose`. Same for `RotationMerge`. The pass `Mapper` requires extra information for running (the `coupling_map`, in this case).

```
pm = PassManager()
pm.add_pass(CxCancellation()) # requires: ToffoliDecompose / preserves: ToffoliDecompose
pm.add_pass(RotationMerge())  # requires: ToffoliDecompose / preserves: ToffoliDecompose
pm.add_pass(Mapper(coupling_map=coupling_map))         # requires: {} / preserves: {}
pm.add_pass(CxCancellation())
```

The sequence of passes to execute in this case is:

1. `ToffoliDecompose`, because it is required by `CxCancellation`.
2. `CxCancellation`
3. `RotationMerge`, because, even when `RotationMerge` also requires `ToffoliDecompose`, the `CxCancellation` preserved it, so no need to run it again.
4. `Mapper`
1. `ToffoliDecompose`, because `Mapper` did not preserved `ToffoliDecompose` and is require by `CxCancellation`
2. `CxCancellation`

### Pass identity
A pass behavior can be heavily influenced by its parameters. For example, unrolling using some basis gates is totally different than unrolling to different gates. And a PassManager might use both.

```
pm.add_pass(Unroller(basis_gates=['cx','id','u0','u1','u2','u3']))
pm.add_pass(...)
pm.add_pass(Unroller(basis_gates=['U','CX']))
```

where (from `qelib1.inc`):

```
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate u1(lambda) q { U(0,0,lambda) q; }
gate cx c,t { CX c,t; }
gate id a { U(0,0,0) a; }
gate u0(gamma) q { U(0,0,0) q; }
```

For this reason, the identity of a pass is given by its name and parameters.

### Fixed point
There are cases when one or more passes have to run until a condition is fulfilled.

```
pm = PassManager()
pm.add_pass([CxCancellation(), HCancellation(), CalculateDepth()],
                do_while=lambda property_set: not property_set.fixed_point('depth'))
```
The control argument `do_while` will run these passes until the callable returns `False`. In this example, `CalculateDepth` is an analysis pass that updates the property `depth` in the property set.

### Conditional 
The pass manager developer can avoid one or more passes by making them conditional (to a property in the property set)

```
pm.add_pass(LayoutMapper(coupling_map))
pm.add_pass(CheckIfMapped(coupling_map))
pm.add_pass(SwapMapper(coupling_map),
                condition=lambda property_set: not property_set['is_mapped'])
``` 

The `CheckIfMapped` is an analysis pass that updates the property `is_mapped`. If `LayoutMapper` could map the circuit to the coupling map, the `SwapMapper` is unnecessary. 

### Missbehaving passes
To help the pass developer discipline, if an analysis pass attempt to modify the dag or if a transformation pass tries to set a property in the property manager, a `TranspilerAccessError` raises.

The enforcement of this does not attempt to be strict.

## Next Steps

 * support "provides". Different mappers provide the same feature. In this way, a pass can request `mapper` and any mapper that provides `mapper` can be used.
* It might be handy to have property set items in the field `requires` and analysis passes that "provide" that field.
* Move passes to this scheme and, on the way, well-define a DAG API.
