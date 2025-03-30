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

    # define the fesible region for the forward problem
    exp_num = 10
    type = 'pre'
    dim_f = 4
    dim_p = 4
    e = np.ones(dim_f)
    g1 = math.sqrt(2)
    noisy = 0

    #generate training data
    rd.seed(3)
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
        [A, b, iter_val] = utili.inverse_gradient_regular(A, iter_num, G, g, dim_f, dim_p, x, c, type)
        end = time.time()
        final_training_loss.append(iter_val[-1])

    print(final_training_loss)


    ###################################################################################


