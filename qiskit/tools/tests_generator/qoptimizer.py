import argparse

import sys
import os
rootFolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(rootFolder)
print rootFolder
import shutil

from extensible_gate_domain import _basic_gates_string_IBM, _basic_gates_string_IBM_advanced
from qoptimizer_cache_executor import CacheExecutor

import plot_cost
import math
import random
from mutator import replaceGate, replaceBits, replaceInstruction, buildRandomInstruction, mutate, check_entanglement
from mutator import unaryOPs, binaryOPs, ternaryOPs
import qoptimizer_cost_estimator
from qoptimizer_cost_estimator import correctness_cost, perf_cost
from equivalence_checker import check_equivalence

# need to refactor the name "basis_gates"

# highlights:
# we do not consider u gates since they are not interpretable or generalizable for humans..
# we want to produce the code that is generalizable


_basic_gates_string = None
def makeArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimization_level', type=int, default=2,
                           help='1: heuristics based merging, 2. stochastic compiler optimization')

    parser.add_argument('--basis_gates_option', type=int, default=2,
                           help='1: ibm q, 2: ibm q advanced')
    parser.add_argument('--qasm_files', type=str, default="./testcases/grover11.qasm", #
                       help='paths of the qasm files') # they are separated with

    parser.add_argument('--return_measured_state', type=int, default=1,
                   help='for math execution, 1 return measured state only, 0 return complete state')

    parser.add_argument('--iterations', type=int, default=1000,
                           help='how many itereations are needed by each mcmc run')

    parser.add_argument('--mcmc_repeat', type=int, default=1,
                           help='how many times do we repeat the mcmc')

    parser.add_argument('--beta', type=float, default=0.05,
                           help='beta is used in describing \pi')

    parser.add_argument('--beta_delta', type=float, default=0.02,
                           help='increase of beta every time')


    parser.add_argument('--beta_period', type=int, default=200,
                           help='how many iterations should update beta after?')

    parser.add_argument('--disorder_penality', type=int, default=1,
                           help='what is the cost of misalignment between two matrix states?')

    parser.add_argument('--correctness_perf_ratio', type=int, default=1,
                           help='ratio of weight of correctness over perf')

    parser.add_argument('--mcmc_folder', type=str, default="./mcmc_mutants", #
                       help='folder for holding mutant files') # they are separated with

    parser.add_argument('--mcmc_cost_logs', type=str, default="./mcmc_cost_logs", #
                       help='folder for holding cost log files') # they are separated with


    args = parser.parse_args()

    global _basic_gates_string
    if args.basis_gates_option == 1:
        _basic_gates_string = _basic_gates_string_IBM
    else:
        _basic_gates_string = _basic_gates_string_IBM_advanced

    return args


# the string can be file1:file2:dir1:dir2
# if it is a dir, we will look for *.qasm inside the dir
def parse_paths_linux_style(qasm_paths_str):
    qasm_paths = qasm_paths_str.split(':')
    result = []
    for qasm_path in qasm_paths:
        if os.path.isdir(qasm_path):
            for root, directories, files in os.walk(qasm_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    if filepath.endswith('.qasm') and filepath not in result:
                        result.append(filepath)  # Add it to the list.
        else:
            if os.path.isfile(qasm_path) and qasm_path.endswith('.qasm') and qasm_path not in result:
                result.append(qasm_path)
    return result




def remove_useless(cexecutor, oracle_file, useless):
    if useless != oracle_file:
        cexecutor.cache_purge(useless)
        if os.path.exists(useless):
            os.remove(useless)


def append_cost_to_file(new_file, new_cost_correctness, new_cost_perf, new_cost):
    with open(new_file, 'a') as newFile:
        newFile.write("// new_cost_correctness: " + str(new_cost_correctness) + "\n")
        newFile.write("// new_cost_perf: " + str(new_cost_perf) + "\n")
        newFile.write("// new_cost: " + str(new_cost) + "\n")




# mcmc
def mcmc(repeatI, args, oracle_file, cexecutor):
    current_file = oracle_file
    current_cost = sys.maxint
    beta = args.beta
    oracle_file_name = os.path.basename(oracle_file)
    cost_log_file_name = os.path.join(args.mcmc_cost_logs, oracle_file_name+".run"+str(repeatI)+ "_ratio"+str(args.correctness_perf_ratio) + "_iterCount"+str(args.iterations)+".cost_log")

    with open(cost_log_file_name, 'w+') as cost_log_file:
        cost_log_file.write("#  iteration_id      cost_correctness     cost_perf       cost_total=ratio*cost_correctness + cost_perf\n")

    for i in range(args.iterations):
        if i%10 == 0:
            print "#completed iterations: ", i


        if i % args.beta_period == 0:
            beta += args.beta_delta

        # proposal: current -> new
        newFileName = oracle_file_name + ".run" + str(repeatI) +  "_iter" + str(i) + "_ratio"+str(args.correctness_perf_ratio) + "_iterCount"+str(args.iterations);

        new_file = os.path.join(args.mcmc_folder, newFileName)
        bitDomain = cexecutor.cache_get(current_file, "bitDomain")

        while True:
            mutate(current_file, bitDomain, new_file, None) # we do not determine whether the mutant file is too long (minimization will take care of it)
            has_entanglement_gate = check_entanglement(new_file)
            if has_entanglement_gate:
                break

        cexecutor.cache_execute(args, new_file) # will be executed anyway

        # accept it or not, this is a question for mcmc
        p = random.uniform(0,1)
        new_cost_correctness = correctness_cost(cexecutor, oracle_file, new_file, args.disorder_penality)
        new_cost_perf = perf_cost(cexecutor, oracle_file, new_file)
        new_cost = args.correctness_perf_ratio * new_cost_correctness + new_cost_perf


        current_cost_correctness = correctness_cost(cexecutor, oracle_file, current_file, args.disorder_penality)
        current_cost_perf = perf_cost(cexecutor, oracle_file, current_file)
        current_cost = args.correctness_perf_ratio * current_cost_correctness + current_cost_perf


        costdiff =  new_cost - current_cost
        acceptP = min(1, math.exp(-beta*costdiff))
        if p<= acceptP: # accept new one, current one becomes useless
            useless = current_file
            current_file = new_file
            current_cost_correctness = new_cost_correctness
            current_cost_perf = new_cost_perf
            current_cost = new_cost

            if new_cost_correctness == 0 and new_cost < 0: # faster but still correct
                print "\n\n\nFound one with zero correctness cost:", new_file, new_cost
                append_cost_to_file(new_file, new_cost_correctness, new_cost_perf, new_cost)
                mcmc_mutants_good = args.mcmc_folder + "_good/"
                shutil.copy(new_file, mcmc_mutants_good)


        else: # new_file is useless, keep current_file
            useless = new_file

        remove_useless(cexecutor, oracle_file, useless)

        with open(cost_log_file_name, 'a+') as cost_log_file:
            cost_log_file.write(str(i) + "    ")
            cost_log_file.write(str(current_cost_correctness) + "    ")
            cost_log_file.write(str(current_cost_perf) + "    ")
            cost_log_file.write(str(current_cost) + "\n")

    # end of run-i:
    plot_cost.drawPDF(cost_log_file_name)
    return current_file, current_cost


def optimize_one_file(args, orig_file):
    global _basic_gates_string
    cexecutor = CacheExecutor(_basic_gates_string)
    cexecutor.cache_execute(args, orig_file)

    minInstance = None
    minCost = sys.maxint
    for i in range(args.mcmc_repeat):
        tmpminInstance, tmpminCost = mcmc(i, args, orig_file, cexecutor)
        if tmpminCost < minCost:
            minCost = tmpminCost
            minInstance = tmpminInstance
    print minInstance, minCost

def qoptimizermain(args):
    qasm_paths_str = args.qasm_files
    result = parse_paths_linux_style(qasm_paths_str)
    for qasm_file in result:
        optimize_one_file(args, qasm_file)



if __name__ == '__main__':
    args = makeArgs()
    qoptimizer_cost_estimator.disorder_penality = args.disorder_penality
    qoptimizer_cost_estimator.correctness_perf_ratio = args.correctness_perf_ratio
    if not os.path.exists(args.mcmc_folder):
        raise SystemError("mcmc folder does not exist, fix this first")
    qoptimizermain(args)

