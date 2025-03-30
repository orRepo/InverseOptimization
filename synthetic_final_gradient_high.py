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

    type = 'pre'
    dim_f = 4
    dim_p = 4
    e = np.ones(dim_f)
    g1 = math.sqrt(2)
    noisy = 0

    #generate training data
    #rd.seed(0)
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
    utili.draw_obj(iter_val)
    utili.draw_obj(iter_res1)
    utili.draw_obj(iter_eps)
    utili.draw_region(x, c, b, A, G, g)

    ###################################################################################
    # evaluate the performance
    test_num = 500
    test_seed = 101

    [test_signal, test_observed, test_true] = utili.gendata1(test_num, dim_f, test_seed, 0, e, g1) # (datanum, d, seeds, noise, A, b)
    d = dim_f
    test_recovered = np.zeros((test_num, d))
    for i in range(test_num):
        test_recovered[i] = utili.forward2A(test_signal[i], b, A, G, g)

    [residual, true_predict] = utili.L_p_true1(test_signal, test_recovered, test_true, e, g1)
    true_subopt = utili.L_sub_true1(test_signal, test_recovered, test_true, e, g1)
    out_predict = utili.L_pA(test_signal, test_observed, G, g, A, b)
    out_subopt = utili.L_subA(test_signal, test_observed, G, g, A, b)

    print(
        f"out_predict={out_predict}, true_predict = {true_predict}, out_subopt={out_subopt}, true_subopt = {true_subopt}.")
    print(iter_val[-1])
    print(f'Running time is {end-now}')

