
# This represents user code

import stevedore


# We should derive from some API (like UnitarySynthesisPlugin) but let's ignore this for now
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.circuit.library import LinearFunction
from qiskit.quantum_info import Clifford, decompose_clifford
from qiskit.transpiler import TranspilerError, PassManager
from qiskit.transpiler.synthesis import cnot_synth
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig, HighLevelSynthesis


class LinearFunctionSynthesisPluginForDepth:
    def run(self, linear_function, **options):
        # This functionality should probably NOT be a part of plugin, but let's use it here as
        # experiment.
        print(f"    -> Running LinearFunctionSynthesisPluginForDepth")
        decomposition = cnot_synth(linear_function.linear)
        if linear_function.original_circuit is not None:
            if decomposition.depth() >= linear_function.original_circuit.depth():
                decomposition = linear_function.original_circuit
        return decomposition


class LinearFunctionSynthesisPluginForCount:
    def _count(self, qc):
        ops = qc.count_ops()
        cnt = ops.get('cx', 0) + 3 * ops.get('swap', 0)
        return cnt

    def run(self, linear_function, **options):
        # This functionality should probably NOT be a part of plugin, but let's use it here as
        # experiment.
        print(f"    -> Running LinearFunctionSynthesisPluginForCount")
        decomposition = cnot_synth(linear_function.linear)
        if linear_function.original_circuit is not None:

            if self._count(decomposition) >= self._count(linear_function.original_circuit):
                decomposition = linear_function.original_circuit
        return decomposition


class CliffordSynthesisPluginMain:
    def run(self, cliff, **options):
        print(f"    -> Running CliffordSynthesisPluginDefault")
        decomposition = decompose_clifford(cliff)
        return decomposition


class CliffordSynthesisPluginSkip:
    def run(self, cliff, **options):
        print(f"    -> Running CliffordSynthesisPluginSkip")
        return None



def test():
    # create a small linear circuit
    qcl = QuantumCircuit(3)
    qcl.cx(0, 1)
    qcl.cx(1, 2)
    qcl.swap(0, 2)
    qcl.swap(1, 2)

    # create a small clifford circuit
    qcc = QuantumCircuit(2)
    qcc.h(0)
    qcc.h(1)
    qcc.cx(0, 1)
    qcc.s(1)

    qc = QuantumCircuit(5)
    qc.append(LinearFunction(qcl), [2, 3, 4])
    qc.barrier()
    qc.append(Clifford(qcc), [1, 2])
    print(qc)

    print("Transpiling without config")
    qct = transpile(qc, optimization_level=1)
    print(qct)
    print("")

    print(f"Creating config & transpile for depth")
    hls_config = HLSConfig(use_default_on_unspecified=True)
    hls_config.set_methods("linear_function", [("depth_opt", {})])
    # hls_config.print()
    qct = transpile(qc, optimization_level=1, hls_config=hls_config)
    print(qct)
    print("")

    print(f"Creating config & transpile for 2q count")
    hls_config = HLSConfig(use_default_on_unspecified=True)
    hls_config.set_methods("linear_function", [("count_opt", {})])
    # hls_config.print()
    qct = transpile(qc, optimization_level=1, hls_config=hls_config)
    print(qct)
    print("")

    print(f"Creating config & transpile for depth (only linear functions)")
    hls_config = HLSConfig(use_default_on_unspecified=False)
    hls_config.set_methods("linear_function", [("depth_opt", {})])
    # hls_config.print()
    qct = transpile(qc, optimization_level=1, hls_config=hls_config)
    print(qct)
    print("")

    print(f"Creating config & transpile for 2q count (only linear functions)")
    hls_config = HLSConfig(use_default_on_unspecified=False)
    hls_config.set_methods("linear_function", [("count_opt", {})])
    # hls_config.print()
    qct = transpile(qc, optimization_level=1, hls_config=hls_config)
    print(qct)
    print("")

    print(f"Creating config & transpile for 2q count (do nothing)")
    hls_config = HLSConfig(use_default_on_unspecified=False)
    # hls_config.print()
    qct = transpile(qc, optimization_level=1, hls_config=hls_config)
    print(qct)
    print("")

    # The code below should raise an error since we don't have method "undefined" in linear function's
    # entry points:
    try:
        print(f"Creating config & transpile for 2q count with undefined method")
        hls_config = HLSConfig(use_default_on_unspecified=True)
        hls_config.set_methods("linear_function", [("undefined", {})])
        # hls_config.print()
        qct = transpile(qc, optimization_level=1, hls_config=hls_config)
        print(qct)
        print("")
    except TranspilerError:
        print(f"  -> Throws an error.")
        pass

    # The code below works, because we never encounter an Operation with name "undefined_op".
    # Is this fine, or is this an error?
    print(f"Creating config & transpile for 2q count (for some undefined gate name)")
    hls_config = HLSConfig(use_default_on_unspecified=True)
    hls_config.set_methods("undefined_op", [("undefined", {})])
    # hls_config.print()
    qct = transpile(qc, optimization_level=1, hls_config=hls_config)
    print(qct)
    print("")

    print(f"Creating config & transpile for depth -- NEW!")
    hls_config = HLSConfig(use_default_on_unspecified=True,
                           linear_function=[("depth_opt", {})],
                           clifford=[])
    hls_config.print()
    qct = transpile(qc, optimization_level=1, hls_config=hls_config)
    print(qct)
    print("")

    print(f"Creating clifford -- NEW!")
    hls_config = HLSConfig(use_default_on_unspecified=True,
                           linear_function=[("depth_opt", {})],
                           )
    hls_config.print()
    qct = transpile(qc, optimization_level=1, hls_config=hls_config)
    print(qct)
    print("")


def test2():
    linear_function_ext_plugins = stevedore.ExtensionManager(
        "qiskit.synthesis.linear_function", invoke_on_load=True, propagate_map_exceptions=True
    )
    print(linear_function_ext_plugins.names())
    print("")

    for method_name in linear_function_ext_plugins.names():
        print(f"{method_name = }")
        print(f"{linear_function_ext_plugins[method_name] = }")
        print(f"{linear_function_ext_plugins[method_name].entry_point = }")
        print(f"{linear_function_ext_plugins[method_name].plugin = }")
        print(f"{linear_function_ext_plugins[method_name].obj = }")
        print("")

    print("---------")
    linear_function_ext_hooks = stevedore.HookManager(
        "qiskit.synthesis.linear_function",
        "depth_opt",
        invoke_on_load=True,
    )

    print(linear_function_ext_hooks.names())
    print("")


    for method_name in linear_function_ext_hooks.names():
        print(f"{method_name = }")
        print(f"{linear_function_ext_hooks[method_name] = }")

        print("")

def test3():

    # create a small linear circuit
    qcl = QuantumCircuit(3)
    qcl.cx(0, 1)
    qcl.cx(1, 2)
    qcl.swap(0, 2)
    qcl.swap(1, 2)

    # create a small clifford circuit
    qcc = QuantumCircuit(2)
    qcc.h(0)
    qcc.h(1)
    qcc.cx(0, 1)
    qcc.s(1)

    qc = QuantumCircuit(5)
    qc.append(LinearFunction(qcl), [2, 3, 4])
    qc.barrier()
    qc.append(Clifford(qcc), [1, 2])
    print(qc)

    print("Transpiling without config")
    qct = PassManager(HighLevelSynthesis()).run(qc)
    print(qct)
    print("")




if __name__ == "__main__":
    test()
    # test2()
    # test3()
