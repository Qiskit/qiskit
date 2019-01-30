## Goals and elements
The main goal of Qiskit Terra's transpiler is to provide an extensible infrastructure of pluggable passes that allows flexibility in customizing the compilation pipeline through the creation and combination of new passes.

### Passes
- Passes run with the implementation of the abstract method `run`, which takes and returns a DAG (directed acyclic graph) representation of the circuit.
- Passes are instances of either `AnalysisPass` or `TransformationPass`.
- Passes are described not just by their class, but also by their parameters (see Use cases: pass identity)
- Analysis passes analyze the DAG and write conclusions to a common context, a `PropertySet` object. They cannot modify the DAG.
- Transformation passes can alter the DAG, but have read-only access to the property set.

### Pass Manager
- A `PassManager` instance determines the schedule for running registered passes.
- The pass manager is in charge of deciding the next pass to run, not the pass itself.
- Registering passes in the pass manager pipeline is done by the `append` method.
- While registering, you can specify basic control primitives over each pass (conditionals and loops).
- Options to control the scheduler:
	- Passes can have arguments at init time that can affect their scheduling. If you want to set properties related to how the pass is run, you can do so by accessing these properties (e.g. pass_.max_iteration = 10).
	- Options set from the pass manager take more precedence over those set at the time of adding a pass set, and those take more precedence over the options of each individual pass. (see [tests](https://github.com/Qiskit/qiskit-terra/master/test/transpiler/test_pass_scheduler.py))


### Pass dependency control
The transpiler architecture allows passes to declare two kinds of dependency control to the pass manager:
- `requires` are passes that need to have been run before executing the current pass.
- `preserves` are passes that are not invalidated by the current pass.
- Analysis passes preserve all.
- The `requires` and `preserves` lists contain concrete instances of other passes (i.e. with specific pass parameters).


### Control Flow Plugins
By default, there are two control flow plugins included in the default pass manager: `do_while` and `conditional` (see **Fixed Point** and **Conditional** sue cases). You might want to add more control flow plugins. For example, a for-loop can be implemented in the following way:

```Python
class DoXTimesController(FlowController):
    def __init__(self, passes, do_x_times, **_):
        self.do_x_times = do_x_times()
        super().__init__(passes)

    def __iter__(self):
        for _ in range(self.do_x_times):
            for pass_ in self.working_list:
                yield pass_
```

The plugin is added to the pass manager in this way:

```
self.passmanager.add_flow_controller('do_x_times', DoXTimesController)
```

This allows to use the parameter `do_x_times`, which needs to be a callable. In this case, this is used to parametrized the plugin, so it will for-loop 3 times.

```
self.passmanager.append([Pass()], do_x_times=lambda x : 3)
```


## Use cases
### A simple chain with dependencies:
The `CxCancellation` requires and preserves `ToffoliDecompose`. Same for `RotationMerge`. The pass `Mapper` requires extra information for running (the `coupling_map`, in this case).

```
pm = PassManager()
pm.append(CxCancellation()) # requires: ToffoliDecompose / preserves: ToffoliDecompose
pm.append(RotationMerge())  # requires: ToffoliDecompose / preserves: ToffoliDecompose
pm.append(Mapper(coupling_map=coupling_map))         # requires: [] / preserves: []
pm.append(CxCancellation())
```

Given the above, the pass manager executes the following sequence of passes:

1. `ToffoliDecompose`, because it is required by `CxCancellation`.
2. `CxCancellation`
3. `RotationMerge`, because, even when `RotationMerge` also requires `ToffoliDecompose`, the `CxCancellation` preserved it, so no need to run it again.
4. `Mapper`
5. `ToffoliDecompose`, because `Mapper` did not preserved `ToffoliDecompose` and is require by `CxCancellation`
6. `CxCancellation`

### Same pass with different parameters (pass identity)
A pass behavior can be heavily influenced by its parameters. For example, unrolling using some basis gates is totally different than unrolling to different gates. And a PassManager might use both.

```
pm.append(Unroller(basis_gates=['id','u1','u2','u3','cx']))
pm.append(...)
pm.append(Unroller(basis_gates=['U','CX']))
```

where (from `qelib1.inc`):

```
gate id q { U(0,0,0) q; }
gate u1(lambda) q { U(0,0,lambda) q; }
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate cx c,t { CX c,t; }
```

For this reason, the identity of a pass is given by its name and parameters.

### While loop up to fixed point
There are cases when one or more passes have to be run repeatedly, until a condition is fulfilled.

```
pm = PassManager()
pm.append([CxCancellation(), RotationMerge(), CalculateDepth()],
              do_while=lambda property_set: not property_set['fixed_point']['depth'])
```
The control argument `do_while` will run these passes until the callable returns `False`. The callable always takes in one argument, the pass manager's property set. In this example, `CalculateDepth` is an analysis pass that updates the property `depth` in the property set.

### Conditional 
The pass manager developer can avoid one or more passes by making them conditional (on a property in the property set):

```
pm.append(BasicLayout(coupling_map))
pm.append(CheckMap(coupling_map))
pm.append(BasicSwap(coupling_map),
              condition=lambda property_set: not property_set['is_swap_mapped'])
``` 

The `CheckMap` is an analysis pass that updates the property `is_swap_mapped`. If `LayoutMapper` could map the circuit to the coupling map, the `SwapMapper` is unnecessary.


### Idempotent passes
If a pass is idempotent, the transpiler can use that property to perform certain optimizations.
A pass is idempotent if `pass.run(pass.run(dag)) == pass.run(dag)`. Analysis passes are idempotent by definition, since they do not modify the DAG. Transformation passes can declare themselves as idempotent by annotating as *self preserve* in the following way (`<-`):

```Python
class IdempotentPass(TransformationPass):
    def __init__(self):
        super().__init__()
        self.preserves.append(self)  # <-
```

### Misbehaving passes
To help the pass developer discipline, if an analysis pass attempts to modify the DAG or if a transformation pass tries to set a property in the property set of the pass manager, a `TranspilerAccessError` raises.

The enforcement of this does not attempt to be strict.

## Next Steps

* Support "provides". Different mappers provide the same feature. In this way, a pass can request `mapper` and any mapper that provides `mapper` can be used.
* It might be handy to have property set items in the field `requires` and analysis passes that "provide" that field.
* Move passes to this scheme and, on the way, well-define a DAG API.
