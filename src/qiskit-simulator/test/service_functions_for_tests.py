#!/usr/bin/python3

'''
This file contains auxiliary functions, to be called from test scripts.
Its purpose is to avoid duplication.
'''

import os
import argparse
import random
import numpy
import qiskit.backends._qiskit_cpp_simulator as qsim


# ** default_executable **
def default_executable():
    '''
    Returns the simulator executable in this Qiskit environment, i.e,
    <qiskit root>/out/qiskit_simulator
    '''
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../out/qiskit_simulator'))


# ** parse **
def parse(description):
    """
    Creates and returns a parser for the test's command line arguments.
    Parameter 'description' should contain a very brief description of the test.
    
    The parser has two arguments: --seed and --executable.

    If argument 'seed' is set, then we initialize both 'random' and 'numpy.random' with this seed.
    Be aware that you have to insert the seed to the simulator's config file.

    Argument 'executable' is the qiskit simulator executable.
    Its default is the executable <qiskit root>/out/qiskit_simulator

    Since 'parse' returns the parser, you can add more arguments.
    For example run_tests adds arguments --reference_executable and --dir.
    """
    
    parser = argparse.ArgumentParser(description = description,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--seed', type = int, dest = 'seed',
                        help = 'random seed')

    parser.add_argument('--executable', type = str, dest = 'executable',
                        help = 'qiskit simulator executable',
                        default = default_executable())
                        
    args = parser.parse_known_args()[0]

    if args.seed is not None:
        random.seed(a = args.seed)
        numpy.random.seed(args.seed)

    return parser


# ** run_simulator **
def run_simulator(qobj, executable):
    """
    Runs the qiskit simulator on object 'qobj' and returns the result.
    In addition, prints the result to the screen if the execution failed.
    Prints 'Successful execution' and run time for successful executions.
    """

    # TODO: add a parameter 'seed' to 'run_simulator'
    # If seed is not None: qobj['config']['seed'] = seed
    # This way, the user will not have to remember to attach the seed to qobj.
    
    result = qsim.run(qobj, executable = executable)

    if result['success'] == True:
        print('Successful execution. Execution time: ' + str(result['time_taken']))
    else:
        print('Execution failed. Details:')
        print(result)

    return result
