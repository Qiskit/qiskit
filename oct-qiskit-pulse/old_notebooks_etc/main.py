#import qUser
#import optimizer
#import pulse_compiler
#import executor



'''first we intake the user input, for starters just a single gate'''

gate = user_input.gate
'sim_backend and exec backend should also be pulled out'
''''''
original_pulse = compiled_pulse = pulse_compiler.compile_pulse_from_gate(gate)

pulse_optimizer = new pulse_optimizer(compiled_pulse, backend, **param)

optimized_pulse = pulse_optimizer.optimize_params(**params)

executor = new pulse_executor(backend, **params)

executor.run(optimized_pulse, shots, **params)
executor.generate_comparison(optimized_pulse, original_pulse)






