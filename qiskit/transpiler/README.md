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
- Transformation passes can alter the DAG.

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
- Passes are described not just by their name, but also by their parameters. This is because, for example, `UnRoller` is different, depending on the `basis_gates` argument. Unrolling using some basis gates is totally different than unrolling to different gates. And a PassManager might use both.



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

### Missbehaving passes
* The enforcement of this does not need to be strict and it's implemented in [`._fencedobjs`](https://github.com/Qiskit/qiskit-terra/compare/master...1ucian0:transpiler?expand=1#diff-4622ec7af357a91db2367fd25ab1ca53)

## Next Steps

 * support "provides". Different mappers provide the same feature. In this way, a pass can request `mapper` and any mapper that provides `mapper` can be used.
* It might be handy to have property set items in the field `requires` and analysis passes that "provide" that field.
* Move passes to this scheme and, on the way, well-define a DAG API.
