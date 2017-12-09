
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def makeArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--iteration_id', type=int, default=-1, #
                       help='paths of the qasm files') # they are separated with

    parser.add_argument('--points_file', type=str, default=None, #
                       help='paths of the qasm files') # they are separated with

    args = parser.parse_args()



    return args


import os

def collectPoints(iteration_id):
    # print cost_log_file
    # filename = os.path.basename(cost_log_file)
    # parts = filename.split(".")
    # figname = parts[0]
    # if len(parts) >=3:
    #     figname = figname + "-run-" + parts[2]

    pointsFileName = os.path.join(".", "iteration="+str(iteration_id) + "_correct.points")
    with open(pointsFileName, "w") as pointsFile:
        for file in os.listdir("."):
            if file.endswith(".cost_log"):
                cost_log_file = os.path.join(".", file)

                if "qubits=" not in cost_log_file:
                    continue
                with open(cost_log_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        if not line.startswith("#") and line != '\n':
                            itid = line.split()[0]
                            correct = line.split()[1]

                            cost = line.split()[3]
                            if int(itid) == iteration_id and int(correct) == 0:
                                pointsFile.write(itid + " " + cost + "\n")


def drawdist(pointsfile):
    arr = []
    with open(pointsfile) as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                itid = line.split()[0]
                cost = line.split()[1]
                arr.append(int(cost))
                if len(arr) > 500:
                    break

    plt.hist(arr)
    axes = plt.gca()
    axes.set_ylim([0,180]) # can be updated
    plt.show()
    plt.savefig(pointsfile + '_dist.pdf')



if __name__ == '__main__':
    args = makeArgs()
    if args.iteration_id != -1:
        collectPoints(args.iteration_id)
    if args.points_file != None:
        drawdist(args.points_file)


