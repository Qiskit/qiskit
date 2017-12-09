
import argparse
import numpy as np
import os

import sys
rootFolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(rootFolder)

def makeArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--testcases', type=str, default="./testcases", #
                       help='paths of the qasm files') # they are separated with

    parser.add_argument('--total_num_host', type=int, default=25, #
                       help='paths of the qasm files') # they are separated with


    args = parser.parse_args()



    return args


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

import platform

if __name__ == '__main__':
    args = makeArgs()
    hostname = platform.node()
    hostname = hostname[4:] # liup1
    hostid = int(hostname)
    result = parse_paths_linux_style(args.testcases)
    total_host_num = args.total_num_host
    sorted_result = sorted(result)
    print "# host is: " + platform.node()
    for i in range(len(sorted_result)):
        if i%total_host_num == hostid:
            print "python qoptimizer.py --iterations 100000 --correctness_perf_ratio 2 --qasm_files " + sorted_result[i] + " > " + sorted_result[i] + "_iterCount100000_ratio2.out" + " 2>&1"
            # print "python qoptimizer.py --iterations 100000 --correctness_perf_ratio 4 --qasm_files " + result[i]
            # print "python qoptimizer.py --iterations 100000 --correctness_perf_ratio 6 --qasm_files " + result[i]
            # print "python qoptimizer.py --iterations 100000 --correctness_perf_ratio 8 --qasm_files " + result[i]
            # print "python qoptimizer.py --iterations 100000 --correctness_perf_ratio 10 --qasm_files " + result[i]

