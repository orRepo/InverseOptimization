import gurobipy as gp
from gurobipy import GRB
import numpy.random as rd
import math
import numpy as np
import inverse_utility as utili
from matplotlib import use as mpl_use
import time

if __name__ == "__main__":
    datanum = 100
    # define the fesible region for the forward problem

    g1 = math.sqrt(2)
    dim_f = 5
    dim_p = 4
    e = np.ones(dim_f)
    noisy = 0

    #generate training data
    rd.seed(0)
    x = np.zeros((datanum, dim_f))
    c = np.zeros((datanum, dim_f))
    for i in range(datanum):
        [buff1, buff2] = utili.synthetic_forward1(dim_f, e, g1)
        for j in range(dim_f):
            c[i][j] = buff2[j]
            if noisy:
                x[i][j] = buff1[j] + np.random.normal(0, 0.5)
            else:
                x[i][j] = buff1[j]

    # Learn feasible region
    record_A = []
    [G, g] = utili.genSimplex(dim_p) # simplex
    now = time.time()
    [A, b] = utili.invPSim(x, c, G, g)
    end = time.time()
    utili.draw_region(x, c, b, A, G, g)

    ###################################################################################
    # evaluate the performance
    test_num = 100
    test_seed = 101
    [test_signal, test_observed, test_true] = utili.gendata1(test_num, dim_f, test_seed, 0, e, g1)
    d=dim_f
    test_recovered = np.zeros((test_num, d))
    for i in range(test_num):
        test_recovered[i] = utili.forward2A(test_signal[i], b, A, G, g)

    [residual, true_predict] = utili.L_p_true1(test_signal, test_recovered, test_true, e, g1)
    true_subopt = utili.L_sub_true1(test_signal, test_recovered, test_true, e, g1)
    out_predict = utili.L_pA(test_signal, test_observed, G, g, A, b)
    out_subopt = utili.L_subA(test_signal, test_observed, G, g, A, b)
    print(
        f"out_predict={out_predict}, true_predict = {true_predict}, out_subopt={out_subopt}, true_subopt = {true_subopt}.")
    print(f'Running time is {end - now}')