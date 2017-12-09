
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def makeArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cost_log', type=str, default="/Users/liup/quantum/qiskit-sdk-py/qiskit/qanalyzer/qoptimizer4grove/mcmc_cost_logs/amplify.qasm.run0_ratio1_iterCount100000.cost_log", #
                       help='paths of the qasm files') # they are separated with


    args = parser.parse_args()



    return args


import os

def drawPDF(cost_log_file):
    # print cost_log_file
    # filename = os.path.basename(cost_log_file)
    # parts = filename.split(".")
    # figname = parts[0]
    # if len(parts) >=3:
    #     figname = figname + "-run-" + parts[2]

    print "drawing for the cost log: ", cost_log_file
    if "qubits=" not in cost_log_file:
        return

    with open(cost_log_file) as f:
        lines = f.readlines()
        X = [line.split()[0] for line in lines if not line.startswith("#")]
        Y = [line.split()[3] for line in lines if not line.startswith("#")]


        plt.plot(X,Y, color="blue", linewidth=2.5, linestyle="-.")


        #plt.xticks( [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        # plt.xticks([1,2,3,4,5], [1,2,3,4,5])
        plt.legend(loc='upper right', frameon=False)
        axes = plt.gca()
        axes.set_xlim([0,50000]) # can be removed
        axes.set_ylim([-50,80]) # can be updated


        # plt.show()
        plt.savefig(cost_log_file + '5080.pdf')

if __name__ == '__main__':
    args = makeArgs()
    cost_log_file = args.cost_log
    drawPDF(cost_log_file)

