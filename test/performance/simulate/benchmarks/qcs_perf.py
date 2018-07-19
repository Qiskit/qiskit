""" QCS Perf is a quantum-software bencmark suite """
import argparse
import sys

from backends.executor import Executor
from application.application_gen import ApplicationGenerator

if sys.version_info < (3, 0):
    raise Exception("Please use Python version 3 or greater.")


def run_from_script(app, app_arg, backend, verify):
    """
    run simulator from qiskit
    """

    q_prog = app.gen_application(app_arg)
    elapsed = backend.run_simulation(q_prog)

    if elapsed < 0:
        print("Execution Failed")
        return

    print(app.name + "," + backend.backend_name + "," + str(app_arg["qubit"]) +
          "," + str(app_arg["depth"]) + "," + str(elapsed), flush=True)

    if verify:
        backend.verify_result(app_arg["depth"], app_arg["qubit"])


def run_benchmark(args, qubit):
    """
    Run simulation by each qasm files
    """
    name = args.name
    backend_name = args.backend
    depth = int(args.depth)
    seed = int(args.seed)
    app_seed = int(args.applicationseed)

    if seed:
        seed = int(seed)

    executor = Executor(backend_name, name, seed)
    backend = executor.get_backend(backend_name)

    app = ApplicationGenerator(app_seed)
    gen_app = app.get_app(name)

    if gen_app:
        app_arg = {"qubit": qubit, "depth": depth, "seed": app_seed}
        run_from_script(gen_app, app_arg, backend, args.verify)

    return True


def print_app_list():
    """
    print application list
    """
    app = ApplicationGenerator(None)
    name_list = app.get_app_name_list()

    for name in name_list:
        print(name)


def parse_args():
    """
    argument parser
    """
    parser = argparse.ArgumentParser(
        description=("Evaluate the performance of \
                     simulator with and prints a report."))

    parser.add_argument('-a', '--name', default='qft', help='benchmark name')
    parser.add_argument('-s', '--start', default='4',
                        help='minimum qubits for evaluation')
    parser.add_argument('-e', '--end', default='0',
                        help='maximum qubits for evaluation')
    parser.add_argument('-d', '--depth', default='5', help='depth')
    parser.add_argument('-b', '--backend',
                        default='local_qasm_simulator', help='backend name')
    parser.add_argument('-sd', '--seed', default=17,
                        help='the initial seed (int)')
    parser.add_argument('-as', '--applicationseed', default=17,
                        help='the seed (int) for application')
    parser.add_argument('-v', '--verify', action='store_true',
                        help='verify simulation results')
    parser.add_argument('-l', '--list', action='store_true',
                        help='show application list')

    return parser.parse_args()


def _main():
    args = parse_args()

    if args.list:
        print_app_list()
        return

    start_qubit = int(args.start)
    end_qubit = int(args.end)

    if not end_qubit:
        end_qubit = start_qubit

    for qubit in range(int(args.start), end_qubit + 1):
        if not run_benchmark(args, qubit):
            break


def main():
    """
    main function
    """
    try:
        _main()
    except KeyboardInterrupt:
        print("Benchmark suite interrupted: exit!")
        sys.exit(1)


if __name__ == "__main__":
    main()
