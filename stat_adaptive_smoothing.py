import gurobipy as gp
from gurobipy import GRB
import numpy.random as rd
import math
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import inverse_utility as utili
from scipy.spatial import ConvexHull
from matplotlib import use as mpl_use
import time


if __name__ == "__main__":
    datanum = 200
    exp_num = 10
    # define the fesible region for the forward problem

    type = 'sub'
    dim_f = 4
    dim_p = 4
    e = np.ones(dim_f)
    g1 = math.sqrt(2)
    noisy = 0

    #generate training data
    rd.seed(0)
    final_training_loss = []

    for i_exp in range(exp_num):
        x = np.zeros((datanum, dim_f))
        c = np.zeros((datanum, dim_f))
        for i in range(datanum):
            [buff1, buff2] = utili.synthetic_forward1(dim_f, e, g1) # forward function for exp2 in synthetic
            for j in range(dim_f):
                c[i][j] = buff2[j]
                if noisy:
                    x[i][j] = buff1[j] + np.random.normal(0, 0.2)
                else:
                    x[i][j] = buff1[j]

        # Learn feasible region
        A = rd.rand(dim_f, dim_p)
        iter_num = 100
        record_A = []
        [G, g] = utili.genSimplex(dim_p)
        now = time.time()
        [A, b, iter_val, iter_res1, iter_res2, iter_eps, epep] = utili.inverse_gradient(A, iter_num, G, g, dim_f, dim_p, x, c, type, eps1=1, eps2=1)
        end = time.time()
        final_training_loss.append(iter_val[-1])

    print(final_training_loss)
    ###################################################################################
    # evaluate the performance


