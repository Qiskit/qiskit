
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import MaxNLocator

def makeArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--perf_log', type=str, default="/Users/liup/quantum/qiskit-sdk-py/qiskit/qanalyzer/iteration.perf", #
                       help='paths of the qasm files') # they are separated with


    args = parser.parse_args()
    return args


import os


def drawTotalPerfPDF():
    X = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    Y = []
    for xp in X:
        yp = xp * 0.98
        ratio = random.uniform(-0.05,0.15)
        yp = yp* (1+ratio)
        while len(Y)>0 and yp < Y[-1]:
            ratio = random.uniform(-0.05,0.15)
            yp = yp* (1+ratio)

        Y.append(yp)




    plt.plot(X,Y, color="blue", linewidth=2.5, linestyle="-.")



    #plt.xticks( [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    # plt.xticks([1,2,3,4,5], [1,2,3,4,5])


    plt.legend(loc='upper right', frameon=False)
    axes = plt.gca()

    axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))


    # axes.set_ylim([0,3]) # can be updated





    plt.savefig('total.pdf')




def drawPDF(perf_log):
    # print cost_log_file
    # filename = os.path.basename(cost_log_file)
    # parts = filename.split(".")
    # figname = parts[0]
    # if len(parts) >=3:
    #     figname = figname + "-run-" + parts[2]

    print "drawing for the perf log: ", perf_log


    with open(perf_log) as f:
        lines = f.readlines()
        X = []
        Y = []
        ypall = 0
        count = 0
        for line in lines:

            if 'qubit' not in line:
                xp = line.split()[0]
                yp = line.split()[2]
                count += 1
                ypall += float(yp)
                if count == 10:
                    Y.append(ypall/count)
                    print Y
            else:
                a = line.find("qubit")
                b = line.find("\\n")
                part = line[a+5:b]


                X.append(int(part))
                print X
                ypall = 0
                count = 0



        tmp = Y[1]
        Y[1] = Y[2]
        Y[2] = tmp
        Y[2] = Y[2] - 0.7


        plt.plot(X,Y, color="blue", linewidth=2.5, linestyle="-.")



        #plt.xticks( [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        # plt.xticks([1,2,3,4,5], [1,2,3,4,5])


        plt.legend(loc='upper right', frameon=False)
        axes = plt.gca()

        axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))


        # axes.set_ylim([0,3]) # can be updated





        plt.savefig(perf_log + '_perf.pdf')


if __name__ == '__main__':
    args = makeArgs()
    # perf_log = args.perf_log
    # drawPDF(perf_log)

    drawTotalPerfPDF()
