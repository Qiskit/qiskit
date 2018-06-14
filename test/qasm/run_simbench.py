""" QSAM-Bench is a quantum-software bencmark suite """
import argparse
import os.path
import sys
import re
import time
import json
import glob
import operator

import qiskit

if sys.version_info < (3, 0):
    raise Exception("Please use Python version 3 or greater.")


def run_benchmark(args, qubit):
    """
    Run simulation by each qasm files
    """
    name = args.name
    backend = args.backend
    depth = int(args.depth)
    seed = args.seed

    if seed:
        seed = int(seed)

    if depth > 0:
        qasm_files = name + "/" + name + "_n" + \
                     str(qubit) + "_d" + str(depth) + "*.qasm"
        pattern1 = name + "_n" + str(qubit) + \
                          "_d" + str(depth) + r"[^0-9]*\.qasm"
        pattern2 = name + "_n" + str(qubit) + "_d" + str(depth) + r"\D.*\.qasm"
    else:
        qasm_files = name + "/" + name + "_n" + str(qubit) + "*.qasm"
        pattern1 = name + "_n" + str(qubit) + r"[^0-9]*\.qasm"
        pattern2 = name + "_n" + str(qubit) + r"\D.*\.qasm"

    qasm_files = glob.glob(qasm_files)

    if not qasm_files:
        raise Exception("No qasm file")

    for qasm in qasm_files:

        ret = None
        if not ((re.search(pattern1, os.path.basename(qasm))) or
                (re.search(pattern2, os.path.basename(qasm)))):
            continue

        q_prog = qiskit.QuantumProgram()

        if backend.startswith("ibmqx"):
            import Qconfig
            q_prog.set_api(Qconfig.APItoken, Qconfig.config['url'])
        elif not backend.startswith("local"):
            raise Exception('only ibmqx or local simulators are supported')

        q_prog.load_qasm_file(qasm, name=name)

        start = time.time()
        ret = q_prog.execute([name], backend=backend, shots=1,
                             max_credits=5, hpc=None,
                             timeout=60*60*24, seed=seed)
        elapsed = time.time() - start

        if not ret.get_circuit_status(0) == "DONE":
            return False

        if backend.startswith("ibmqx"):
            elapsed = ret.get_data(name)["time"]

        print(name + "," + backend + "," + str(qubit) +
              "," + str(depth) + "," + str(elapsed), flush=True)

        if args.verify:
            verify_result(ret, name, qasm)

    if not ret:
        raise Exception("No qasm file")

    return True


def verify_result(ret, name, qasm):
    """
    Check simulation results
    """

    if not os.path.exists(name + "/ref"):
        raise Exception("Verification not support for " + name)

    ref_file_name = name + "/ref/" + os.path.basename(qasm)+".ref"
    if not os.path.exists(ref_file_name):
        raise Exception("Reference file not exist: " + ref_file_name)

    ref_file = open(ref_file_name)
    ref_data = ref_file.read()
    ref_file.close()
    ref_data = json.loads(ref_data)
    sim_result = ret.get_counts(name)

    sim_result_keys = sim_result.keys()

    for key in sim_result_keys:
        if key not in ref_data:
            raise Exception(key + " not exist in " + ref_file_name)
        ref_count = ref_data[key]
        count = sim_result[key]

        if ref_count != count:
            raise Exception(" Count is differ: " + str(count) +
                            " and " + str(ref_count))


def print_qasm_sum(dir_name):
    """
    List qasm files
    """

    if not os.path.exists(dir_name):
        raise Exception("Not find :" + dir_name)

    file_list = glob.glob(dir_name + "/*.qasm")
    qasm_list = []

    for each_file in file_list:
        file_name = os.path.basename(each_file)
        match_q = re.search("_n([0-9]*)", file_name)
        match_d = re.search("n[0-9]*_d([0-9]*)", file_name)

        if not match_q:
            raise Exception("Not find file:" + dir_name)
        qubit = int(match_q.group(1))

        val = filter(lambda bit: bit['qubit'] == qubit, qasm_list)
        val_list = list(val)

        if not len(val_list):
            if match_d:
                depth = int(match_d.group(1))
                qasm_list.append({"qubit": qubit, "depth": depth, "count": 1})
            else:
                qasm_list.append({"qubit": qubit, "count": 1})
        else:
            if match_d:
                depth = int(match_d.group(1))
                depth_val = list(filter(lambda dep:
                                        dep["depth"] == depth, val_list))
                if not len(depth_val):
                    qasm_list.append({"qubit": qubit,
                                      "depth": depth, "count": 1})
                else:
                    depth_val[0]["count"] += 1
            else:
                val_list[0]["count"] += 1

    if "depth" in qasm_list[0]:
        tmp_list = sorted(qasm_list, key=operator.itemgetter("qubit", "depth"))
    else:
        tmp_list = sorted(qasm_list, key=operator.itemgetter("qubit"))

    print("Application : " + dir_name)
    for each_list in tmp_list:
        print_line = "qubit : " + str(each_list["qubit"])
        if "depth" in each_list:
            print_line += " \t  depth : " + str(each_list["depth"])

        print_line += " \t  file : "+str(each_list["count"])

        print(print_line)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Evaluate the performance of \
                     simulator with and prints a report."))

    parser.add_argument('-a', '--name', default='qft', help='benchmark name')
    parser.add_argument('-s', '--start', default='4',
                        help='minimum qubits for evaluation')
    parser.add_argument('-e', '--end', default='0',
                        help='maximum qubits for evaluation')
    parser.add_argument('-d', '--depth', default='0', help='depth')
    parser.add_argument('-b', '--backend',
                        default='local_qasm_simulator', help='backend name')
    parser.add_argument('-sd', '--seed', default=None,
                        help='the initial seed (int)')
    parser.add_argument('-v', '--verify', action='store_true',
                        help='verify simulation results')
    parser.add_argument('-l', '--list', action='store_true',
                        help='show qasm file')

    return parser.parse_args()


def _main():
    args = parse_args()

    if args.list:
        print_qasm_sum(args.name)
        return

    start_qubit = int(args.start)
    end_qubit = int(args.end)

    if not end_qubit:
        end_qubit = start_qubit

    for qubit in range(int(args.start), end_qubit + 1):
        if not run_benchmark(args, qubit):
            break


def main():
    try:
        _main()
    except KeyboardInterrupt:
        print("Benchmark suite interrupted: exit!")
        sys.exit(1)


if __name__ == "__main__":
    main()
