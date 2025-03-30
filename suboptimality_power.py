
import gurobipy as gp
from gurobipy import GRB
import numpy.random as rd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import time


def inverse(price, g, demand, dim_p, initt):
    m = gp.Model("qp")
    m.setParam('TimeLimit', 2)
    M = 100
    dim_f = len(price[0])
    n = len(demand)
    dim_demand = len(demand[0])
    # B, b, y, v, r

    y = {}
    for i in range(n):
        y[i] = {}
        for j in range(dim_f):
            y[i][j] = m.addVar(lb=-GRB.INFINITY)

    b = {}
    for k in range(dim_p):
        b[k] = {}
        for j in range(dim_f):
            b[k][j] = m.addVar(lb=-GRB.INFINITY)

    B = {}
    for k in range(dim_p):
        B[k] = {}
        for i in range(dim_f):
            B[k][i] = {}
            for j in range(dim_demand):
                B[k][i][j] = m.addVar(lb=-GRB.INFINITY)

    v = {}
    for i in range(n):
        v[i] = {}
        for k in range(dim_p):
            v[i][k] = {}
            for j in range(dim_f):
                v[i][k][j] = m.addVar(lb=-GRB.INFINITY)

    r = {}
    for i in range(n):
        r[i] = {}
        for k in range(dim_p):
            r[i][k] = m.addVar(vtype=GRB.BINARY)

    m.setObjective(1 / 2 * gp.quicksum((g[i][j] - y[i][j]) ** 2 for i in range(n) for j in range(dim_f)), GRB.MINIMIZE)

    m.addConstrs(y[i][j] == gp.quicksum(v[i][k][j] for k in range(dim_p)) for i in range(n) for j in range(dim_f))

    m.addConstrs(v[i][k][j] <= M * r[i][k] for i in range(n) for j in range(dim_f) for k in range(dim_p))
    m.addConstrs(v[i][k][j] >= -M * r[i][k] for i in range(n) for j in range(dim_f) for k in range(dim_p))
    m.addConstrs(v[i][k][j] <= b[k][j] + gp.quicksum(B[k][j][kk] * demand[i][kk] for kk in range(dim_demand))
                 + M * (1 - r[i][k]) for i in range(n) for k in range(dim_p) for j in range(dim_f))
    m.addConstrs(
        v[i][k][j] >= b[k][j] + gp.quicksum(B[k][j][kk] * demand[i][kk] for kk in range(dim_demand)) - M * (1 - r[i][k])
        for i in range(n) for k in range(dim_p) for j in range(dim_f))

    m.addConstrs(gp.quicksum(price[i][j] * y[i][j] for j in range(dim_f))
                 <= gp.quicksum(
        price[i][j] * (b[k][j] + gp.quicksum(B[k][j][kk] * demand[i][kk] for kk in range(dim_demand))) for j in
        range(dim_f))
                 for k in range(dim_p) for i in range(n))

    m.addConstrs(gp.quicksum(r[i][k] for k in range(dim_p)) == 1 for i in range(n))

    for i in range(n):
        r[i][initt[i]].start = 1
    m.optimize()

    opt_b = np.zeros((dim_p, dim_f))
    opt_B = np.zeros((dim_p, dim_f, dim_demand))
    optval = m.objVal
    optr = np.zeros((n, dim_p))

    for i in range(dim_p):
        for j in range(dim_f):
            opt_b[i][j] = b[i][j].X
    for k in range(dim_p):
        for i in range(dim_f):
            for j in range(dim_demand):
                opt_B[k][i][j] = B[k][i][j].X
    for i in range(n):
        for j in range(dim_p):
            optr[i][j] = r[i][j].X

    return [opt_b, opt_B, optr, optval]


def Knownconstrs(opt_g):
    n = len(opt_g[0])
    ak = np.zeros((2 * n, n))
    gk = np.zeros(2 * n)
    for i in range(n):
        ak[i][i] = -1
        gk[i] = 0
    for i in range(n):
        ak[i + n][i] = 1
        gk[i + n] = 3.5

    return [ak, gk]


# [c,optx, G, g] = forward()
def powersys(xiv, d, noise):
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)

    # d=[1,1,1,1,1] #demand at locations

    m = len(xiv)
    g = {}
    for j in range(m):
        g[j] = M1.addVar()

    f = {}
    n = 5  # num of the arcs
    for j in range(n):
        f[j] = M1.addVar(lb=-3.5)

    M1.setObjective(gp.quicksum(xiv[j] * g[j] for j in range(m)), GRB.MINIMIZE)

    M1.addConstrs(g[j] <= 3.5 for j in range(m))
    M1.addConstrs(f[j] <= 3.5 for j in range(n))

    M1.addConstr(g[0] + f[0] - f[1] >= d[0])
    M1.addConstr(f[1] + f[3] >= d[1])
    M1.addConstr(g[1] - f[0] - f[2] >= d[2])
    M1.addConstr(f[2] + f[4] - f[3] >= d[3])
    M1.addConstr(g[2] - f[4] >= d[4])

    M1.optimize()

    opt_g = np.zeros(m)
    for j in range(m):
        opt_g[j] = g[j].X

    if noise:
        for j in range(m):
            opt_g[j] = opt_g[j] + rd.normal(0, 0.2)
            # opt_g[j] = opt_g[j]+rd.laplace(0,0.1/math.sqrt(2))

    return [opt_g, M1.objVal]


def VmM(v, M):
    ans = np.zeros((len(M[0])))

    for i in range(len(M[0])):
        ans[i] = sum(v[j] * M[j][i] for j in range(len(v)))
    return ans


def MmV(M, v):
    ans = np.zeros(len(M))

    for i in range(len(M)):
        ans[i] = sum(M[i][j] * v[j] for j in range(len(v)))
    return ans


def draw_obj(iter_val):
    plt.plot(iter_val, 'b-')
    plt.show()


def numUnique(input_list):
    # taking an input list
    l1 = {}

    count = 0

    for item in input_list:
        if 'item' not in l1:
            count += 1
            l1['item'] = 1
    return count


def forward2(c, demand, b, A, C, Cb, G, g, ak, gk):
    N = len(G)
    m = len(c)
    n = len(G[0])

    M1 = gp.Model("lp1")
    M1.setParam('TimeLimit', 360)

    x = {}
    for j in range(m):
        x[j] = M1.addVar(lb=-GRB.INFINITY)
    z = {}
    for j in range(n):
        z[j] = M1.addVar(lb=-GRB.INFINITY)

    M1.setObjective(gp.quicksum(c[j] * x[j] for j in range(m)), GRB.MINIMIZE)
    M1.addConstrs(gp.quicksum(G[i][j] * z[j] for j in range(n)) <= g[i] for i in range(N))
    M1.addConstrs(x[i] == gp.quicksum(
        (A[i][j] + sum(C[k][i][j] * demand[k] for k in range(len(demand)))) * z[j] for j in range(n))
                  + b[i] + gp.quicksum(Cb[i][k] * demand[k] for k in range(len(demand)))
                  for i in range(m))
    M1.addConstrs(gp.quicksum(ak[j][i] * x[i] for i in range(m)) <= gk[j] for j in range(len(gk)))

    M1.optimize()
    optx = np.zeros(m)
    optz = np.zeros(n)
    for j in range(m):
        optx[j] = x[j].X
    for j in range(n):
        optz[j] = z[j].X
    return [optx, optz]


def Evaluate(A, C, b, x, c, G, g, ak, gk, demand):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G)  # number of the constraints of the primitive set

    d3 = len(demand[0])

    n = len(x)  # data num

    m = gp.Model("main")
    m.setParam('TimeLimit', 3600)

    gammao = {}
    for i in range(n):
        gammao[i] = {}
        for j in range(d1):
            gammao[i][j] = m.addVar(lb=-GRB.INFINITY)

    gammaf = {}
    for i in range(n):
        gammaf[i] = {}
        for j in range(d1):
            gammaf[i][j] = m.addVar(lb=-GRB.INFINITY)

    z = {}
    for i in range(n):
        z[i] = {}
        for j in range(d):
            z[i][j] = m.addVar(lb=-GRB.INFINITY)

    lamb = {}
    for i in range(n):
        lamb[i] = {}
        for j in range(d2):
            lamb[i][j] = m.addVar()

    m.setObjective(1 / n * gp.quicksum(gammaf[i][j] ** 2 + gammao[i][j] ** 2 for i in range(n) for j in range(d1)),
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))

    m.addConstrs(gammao[i][j] == gammaf[i][j] for i in range(n) for j in range(d1))

    m.addConstrs(x[i][j] + gammaf[i][j] == gp.quicksum(
        (A[j][k] + gp.quicksum(demand[i][m] * C[m][j][k] for m in range(d3))) * z[i][k]
        for k in range(d)) + b[j] for i in range(n) for j in range(d1))
    m.addConstrs(gp.quicksum(c[i][j] * (A[j][k] + gp.quicksum(demand[i][m] * C[m][j][k] for m in range(d3)))
                             for j in range(d1)) + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0 for i in
                 range(n) for k in range(d))
    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] + gammao[i][j] - b[j]) for j in range(d1)) + gp.quicksum(
        lamb[i][j] * g[j] for j in range(d2)) <= 0
                 for i in range(n))

    m.optimize()

    return m.objVal

def convexf(x, c, G, g, demand):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G)  # number of the constraints of the primitive set
    d3 = len(demand[0])
    n = len(x)  # data num

    m = gp.Model("main")
    m.setParam('TimeLimit', 360)

    alpha = m.addVar()

    gammaf = {}
    for i in range(n):
        gammaf[i] = {}
        for j in range(d1):
            gammaf[i][j] = m.addVar(lb=-GRB.INFINITY)

    z = {}
    for i in range(n):
        z[i] = {}
        for j in range(d):
            z[i][j] = m.addVar()

    lamb = {}
    for i in range(n):
        lamb[i] = {}
        for j in range(d2):
            lamb[i][j] = m.addVar()
    Cb = {}
    for i in range(d1):
        Cb[i] = {}
        for j in range(d3):
            Cb[i][j] = m.addVar(lb=-GRB.INFINITY)
    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)

    m.setObjective(1/n*gp.quicksum(gammaf[i][j]**2 for i in range(n) for j in range(d1)),
                   GRB.MINIMIZE)

    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] + gammaf[i][j] - b[j] - gp.quicksum(Cb[j][k] * demand[i][k] for k in range(d3))- z[i][j] == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(c[i][k] * alpha + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - alpha * g[i] <= 0 for i in range(d2) for k in range(n))
    # m.addConstrs(gammao[i][j] == gammaf[i][j] for i in range(n) for j in range(d1))
    m.addConstrs(
        gp.quicksum(c[i][j] * (x[i][j] + gammaf[i][j] - b[j] - gp.quicksum(Cb[j][k] * demand[i][k] for k in range(d3)))
                    for j in range(d1)) + gp.quicksum(lamb[i][j] * g[j] for j in range(d2))
        <= 0 for i in range(n))

    m.optimize()

    opt_alpha = alpha.X
    optval = m.objVal
    opt_b = np.zeros(d1)
    optCb = np.zeros((d1, d3))
    # true_opt = 1/2*sum((x[i][j]-y[i][j].X)**2 for i in range(n) for j in range(d1))
    for i in range(d1):
        opt_b[i] = b[i].X
        for j in range(d3):
            optCb[i][j] = Cb[i][j].X

    return [opt_alpha, optval, opt_b, optCb]

def invfixAReg2(A, C, x, c, eps1, eps2, G, g, ak, gk, demand, fixCb):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G)  # number of the constraints of the primitive set

    d3 = len(demand[0])

    n = len(x)  # data num
    known_num = len(ak)  # number of known constraints

    m = gp.Model("main")
    m.setParam('TimeLimit', 360)

    gammao = {}
    for i in range(n):
        gammao[i] = m.addVar()

    gammaf = {}
    for i in range(n):
        gammaf[i] = {}
        for j in range(d1):
            gammaf[i][j] = m.addVar(lb=-GRB.INFINITY)

    gammanew = {}
    for i in range(n):
        gammanew[i] = {}
        for j in range(d):
            gammanew[i][j] = m.addVar(lb=-GRB.INFINITY)

    Cb = {}
    for i in range(d1):
        Cb[i] = {}
        for j in range(d3):
            Cb[i][j] = m.addVar(lb=-GRB.INFINITY)

    z = {}
    for i in range(n):
        z[i] = {}
        for j in range(d):
            z[i][j] = m.addVar()

    lamb = {}
    for i in range(n):
        lamb[i] = {}
        for j in range(d2):
            lamb[i][j] = m.addVar()

    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)

    gkp = {}
    for i in range(n):
        gkp[i] = {}
        for j in range(known_num):
            gkp[i][j] = m.addVar()

    m.setObjective(1/n*gp.quicksum(gammaf[i][j]**2 for i in range(n) for j in range(d1))
                   + 1/n*gp.quicksum(gammao[i]**2 for i in range(n))
                   + eps1/n*gp.quicksum(gammanew[i][j]**2 for i in range(n) for j in range(d)),
                   GRB.MINIMIZE)
    if fixCb:
        m.addConstrs(Cb[i][j] == 0 for i in range(d1) for j in range(d3))

    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] + gammaf[i][j] - b[j] - gp.quicksum(Cb[j][k] * demand[i][k] for k in range(d3))
                - gp.quicksum((A[j][k] + gp.quicksum(demand[i][m] * C[m][j][k] for m in range(d3))) * z[i][k] for k in
                              range(d)) == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(
                gp.quicksum(c[i][j] * (A[j][k] + gp.quicksum(demand[i][m] * C[m][j][k] for m in range(d3)))
                            for j in range(d1)) + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) + gammanew[i][k] == 0)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    # m.addConstrs(gammao[i][j] == gammaf[i][j] for i in range(n) for j in range(d1))
    m.addConstrs(
        gp.quicksum(c[i][j] * (x[i][j] - b[j] - gp.quicksum(Cb[j][k] * demand[i][k] for k in range(d3)))
                    for j in range(d1)) + gp.quicksum(lamb[i][j] * g[j] for j in range(d2))
        <= gammao[i] for i in range(n))

    m.optimize()

    opt_beta = np.zeros((n, d1))
    opt_mu = np.zeros((n, d))
    opt_z = np.zeros((n, d))
    optval = m.objVal
    opt_b = np.zeros(d1)
    optCb = np.zeros((d1, d3))
    # true_opt = 1/2*sum((x[i][j]-y[i][j].X)**2 for i in range(n) for j in range(d1))
    for i in range(d1):
        opt_b[i] = b[i].X
        for j in range(d3):
            optCb[i][j] = Cb[i][j].X

    for i in range(n):
        for k in range(d):
            opt_z[i][k] = z[i][k].X
            opt_mu[i][k] = mu[i][k].Pi
        for j in range(d1):
            opt_beta[i][j] = beta[i][j].Pi

    res1 = sum(gammanew[i][j].X ** 2 for i in range(n) for j in range(d))

    return [opt_beta, opt_mu, opt_z, optval, opt_b, optCb, res1, 0]


def FindAC(b, B, dim_p, dim_f):
    A = np.zeros((dim_f, dim_p))
    C = np.zeros((5, dim_f, dim_p))

    for i in range(5):
        for j in range(dim_f):
            for k in range(dim_p):
                C[i][j][k] = B[k][j][i]
    for i in range(dim_f):
        for j in range(dim_p):
            A[i][j] = b[j][i]

    return [A, C]


def initialMIP(opt, dim_p):
    kmeans = KMeans(n_clusters=dim_p, random_state=0).fit(opt)

    return kmeans.labels_


def violations(x, d):
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)

    n = len(x)
    m = 3  # number of power plants
    R = 5  # number of demand locations

    g = {}
    for j in range(n):
        g[j] = M1.addVar()

    g1 = {}
    for j in range(n):
        g1[j] = {}
        for i in range(m):
            g1[j][i] = M1.addVar()

    g2 = {}
    for j in range(n):
        g2[j] = {}
        for i in range(m):
            g2[j][i] = M1.addVar()

    g3 = {}
    for j in range(n):
        g3[j] = {}
        for i in range(R):
            g3[j][i] = M1.addVar()

    f = {}
    for i in range(n):
        f[i] = {}
        for j in range(5):
            f[i][j] = M1.addVar(lb=-3.5)

    M1.setObjective(gp.quicksum(g[j] for j in range(n)), GRB.MINIMIZE)

    M1.addConstrs(3.5 + g2[t][j] >= x[t][j] for j in range(m) for t in range(n))
    M1.addConstrs(g1[t][j] >= -x[t][j] for j in range(m) for t in range(n))

    M1.addConstrs(f[t][j] <= 3.5 for t in range(n) for j in range(R))

    M1.addConstrs(x[t][0] + f[t][0] - f[t][1] >= d[t][0] - g3[t][0] for t in range(n))
    M1.addConstrs(f[t][1] + f[t][3] >= d[t][1] - g3[t][1] for t in range(n))
    M1.addConstrs(x[t][1] - f[t][0] - f[t][2] >= d[t][2] - g3[t][2] for t in range(n))
    M1.addConstrs(f[t][2] + f[t][4] - f[t][3] >= d[t][3] - g3[t][3] for t in range(n))
    M1.addConstrs(x[t][2] - f[t][4] >= d[t][4] - g3[t][4] for t in range(n))

    M1.addConstrs(g[t] >= g1[t][j] for t in range(n) for j in range(m))
    M1.addConstrs(g[t] >= g2[t][j] for t in range(n) for j in range(m))
    M1.addConstrs(g[t] >= g3[t][j] for t in range(n) for j in range(R))

    M1.optimize()

    optg = np.zeros(n)
    for i in range(n):
        optg[i] = g[i].X

    return optg


def cL_p_true(c, d, x, xtrue):  # Loss function: true predictability loss, H,h true feasible region
    # x: recovered optimal; x_true: true optimal
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n = len(x)  # number of data points
    m = len(x[0])  # dimension of the decision variable
    gamma = {}
    xx = {}
    f = {}
    for i in range(n):
        f[i] = {}
        for j in range(5):
            f[i][j] = M1.addVar(lb=-3.5, ub=3.5)
    for j in range(n):
        gamma[j] = {}
        xx[j] = {}
        for i in range(m):
            gamma[j][i] = M1.addVar(lb=-GRB.INFINITY)
            xx[j][i] = M1.addVar(lb=-GRB.INFINITY)
    M1.setObjective(gp.quicksum(gamma[j][i] * gamma[j][i] for j in range(n) for i in range(m)), GRB.MINIMIZE)

    M1.addConstrs(x[i][j] + gamma[i][j] == xx[i][j] for i in range(n) for j in range(m))
    M1.addConstrs(xx[i][j] <= 3.5 for i in range(n) for j in range(m))
    M1.addConstrs(xx[i][j] >= 0 for i in range(n) for j in range(m))
    # M1.addConstrs(sum(H[i][j]*xx[k][j] for j in range(m)) <= h[i] for i in range(len(h)) for k in range(n))
    M1.addConstrs(xx[i][0] + f[i][0] - f[i][1] >= d[i][0] for i in range(n))
    M1.addConstrs(f[i][1] + f[i][3] >= d[i][1] for i in range(n))
    M1.addConstrs(xx[i][1] - f[i][0] - f[i][2] >= d[i][2] for i in range(n))
    M1.addConstrs(f[i][2] + f[i][4] - f[i][3] >= d[i][3] for i in range(n))
    M1.addConstrs(xx[i][2] - f[i][4] >= d[i][4] for i in range(n))

    M1.addConstrs(
        sum(c[i][j] * xx[i][j] for j in range(m)) - sum(c[i][j] * xtrue[i][j] for j in range(m)) <= 0 for i in range(n))

    M1.optimize()

    residual = {}
    for i in range(n):
        residual[i] = np.zeros(m)
        for j in range(m):
            residual[i][j] = gamma[i][j].X
    return [residual, M1.objVal / n]


def cL_sub_true(c, d, x, xtrue):  # true sub-optimality loss
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n = len(x)  # number of data points
    m = len(x[0])  # dimension of the decision variable
    gammaf = {}
    gammao = {}
    f = {}
    xx = {}
    for i in range(n):
        f[i] = {}
        for j in range(5):
            f[i][j] = M1.addVar(lb=-3.5, ub=3.5)
    for j in range(n):
        gammaf[j] = {}
        gammao[j] = M1.addVar()
        xx[j] = {}
        for i in range(m):
            xx[j][i] = M1.addVar(lb=-GRB.INFINITY)
            gammaf[j][i] = M1.addVar(lb=-GRB.INFINITY)
    M1.setObjective(gp.quicksum(gammaf[j][i] * gammaf[j][i] for j in range(n) for i in range(m))
                    + gp.quicksum(gammao[j] * gammao[j] for j in range(n)), GRB.MINIMIZE)

    M1.addConstrs(xx[i][j] <= 3.5 for i in range(n) for j in range(m))
    M1.addConstrs(xx[i][j] >= 0 for i in range(n) for j in range(m))
    M1.addConstrs(x[i][j] + gammaf[i][j] == xx[i][j] for i in range(n) for j in range(m))
    # M1.addConstrs(sum(H[i][j]*xx[k][j] for j in range(m)) <= h[i] for i in range(len(h)) for k in range(n))
    M1.addConstrs(xx[i][0] + f[i][0] - f[i][1] >= d[i][0] for i in range(n))
    M1.addConstrs(f[i][1] + f[i][3] >= d[i][1] for i in range(n))
    M1.addConstrs(xx[i][1] - f[i][0] - f[i][2] >= d[i][2] for i in range(n))
    M1.addConstrs(f[i][2] + f[i][4] - f[i][3] >= d[i][3] for i in range(n))
    M1.addConstrs(xx[i][2] - f[i][4] >= d[i][4] for i in range(n))
    M1.addConstrs(float(sum(c[i][j] * x[i][j] for j in range(m)) - sum(c[i][j] * xtrue[i][j] for j in range(m)))
                  <= gammao[i] for i in range(n))

    M1.optimize()
    return M1.objVal / n


def cL_pA(x, c, G, g, dimC, delta, C, A, Cb, b):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G)  # num of constraints
    n = len(x)  # data num
    nL = dimC
    m = gp.Model("main")
    m.setParam('TimeLimit', 3600)

    gammao = {}
    for i in range(n):
        gammao[i] = {}
        for j in range(d1):
            gammao[i][j] = m.addVar(lb=-GRB.INFINITY)
    gammaf = {}
    for i in range(n):
        gammaf[i] = {}
        for j in range(d1):
            gammaf[i][j] = m.addVar(lb=-GRB.INFINITY)
    # b = {}
    # for i in range(d1):
    #     b[i] = m.addVar(lb=-GRB.INFINITY)
    z = {}
    for i in range(n):
        z[i] = {}
        for j in range(d):
            z[i][j] = m.addVar(lb=-GRB.INFINITY)
    lamb = {}
    for i in range(n):
        lamb[i] = {}
        for j in range(d2):
            lamb[i][j] = m.addVar()

    m.setObjective(1 / n * gp.quicksum(gammaf[i][j] ** 2 for i in range(n) for j in range(d1)),
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    # m.addConstrs(gp.quicksum(z[i][j] for j in range(d)) == 1 for i in range(n))
    m.addConstrs(gammaf[i][j] == gammao[i][j] for i in range(n) for j in range(d1))

    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(
                    (A[j][k] + gp.quicksum(C[r][j][k] * delta[i][r] for r in range(nL))) * z[i][k] for k in range(d))
                - b[j] - gp.quicksum(Cb[j][k] * delta[i][k] for k in range(nL)) + gammaf[i][j] == 0)
    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(
                gp.quicksum(c[i][j] * (A[j][k] + gp.quicksum(C[r][j][k] * delta[i][r] for r in range(nL)))
                            for j in range(d1)) + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(
        gp.quicksum(c[i][j] * (x[i][j] + gammao[i][j] - b[j] - gp.quicksum(Cb[j][k] * delta[i][k] for k in range(nL)))
                    for j in range(d1)) +
        gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= 0 for i in range(n))

    m.optimize()

    return m.objVal


def cL_sA(x, c, G, g, dimC, delta, C, A, Cb, b):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G)  # num of constraints
    n = len(x)  # data num
    nL = dimC
    m = gp.Model("main")
    m.setParam('TimeLimit', 3600)

    gammao = {}
    for i in range(n):
        gammao[i] = m.addVar()
    gammaf = {}
    for i in range(n):
        gammaf[i] = {}
        for j in range(d1):
            gammaf[i][j] = m.addVar(lb=-GRB.INFINITY)
    # b = {}
    # for i in range(d1):
    #     b[i] = m.addVar(lb=-GRB.INFINITY)
    z = {}
    for i in range(n):
        z[i] = {}
        for j in range(d):
            z[i][j] = m.addVar(lb=-GRB.INFINITY)
    lamb = {}
    for i in range(n):
        lamb[i] = {}
        for j in range(d2):
            lamb[i][j] = m.addVar()

    m.setObjective(1 / n * gp.quicksum(gammaf[i][j] ** 2 for i in range(n) for j in range(d1)) + 1 / n * gp.quicksum(
        gammao[i] ** 2 for i in range(n)), GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))

    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(
                    (A[j][k] + gp.quicksum(C[r][j][k] * delta[i][r] for r in range(nL))) * z[i][k] for k in range(d))
                - b[j] - gp.quicksum(Cb[j][k] * delta[i][k] for k in range(nL)) + gammaf[i][j] == 0)
    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(
                gp.quicksum(c[i][j] * (A[j][k] + gp.quicksum(C[r][j][k] * delta[i][r] for r in range(nL)))
                            for j in range(d1)) + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] - b[j] - gp.quicksum(Cb[j][k] * delta[i][k] for k in range(nL)))
                             for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= gammao[i] for i in range(n))
    m.optimize()
    return m.objVal


if __name__ == "__main__":

    datanum = 100
    dim_f = 3
    dim_p = 6
    rd.seed(10)

    noise = 0

    fixdeman = 0  # 1: fixed demand for debugging
    warmstart = 1
    needdataset = 1
    fixCb = 0

    if (needdataset):
        fixdemand = [1, 1, 1, 1, 1]
        xiv = []
        opt_g = []
        demand = []
        for n in range(datanum):
            buffer = []
            buffer1 = []
            for j in range(dim_f):
                buffer.append(rd.uniform(0.2, 1))
            buffer1.append(rd.uniform(0.3, 1.5))
            buffer1.append(rd.uniform(0.36, 1.8))
            buffer1.append(rd.uniform(0.42, 2.1))
            buffer1.append(rd.uniform(0.48, 2.4))
            buffer1.append(rd.uniform(0.54, 2.7))

            xiv.append(buffer)
            if (fixdeman):
                [buffer2, buffer3] = powersys(buffer, fixdemand, noise)
                opt_g.append(buffer2)
                demand.append(fixdemand)
            else:
                [buffer2, buffer3] = powersys(buffer, buffer1, noise)
                opt_g.append(buffer2)
                demand.append(buffer1)
        # test sets
        rd.seed(0)
        test_price = []
        test_demand = []
        test_g = []
        test_obj = []
        test_num = 200
        for n in range(test_num):
            buffer = []
            buffer1 = []
            for j in range(dim_f):
                buffer.append(rd.uniform(0.2, 1))
            buffer1.append(rd.uniform(0.3, 1.5))
            buffer1.append(rd.uniform(0.36, 1.8))
            buffer1.append(rd.uniform(0.42, 2.1))
            buffer1.append(rd.uniform(0.48, 2.4))
            buffer1.append(rd.uniform(0.54, 2.7))
            test_price.append(buffer)

            if (fixdeman):
                test_demand.append(fixdemand)
                [gg, obj] = powersys(buffer, fixdemand, noise)
                test_g.append(gg)
                test_obj.append(obj)
            else:
                test_demand.append(buffer1)
                [gg, obj] = powersys(buffer, buffer1, noise)
                test_g.append(gg)
                test_obj.append(obj)

    # define the primitive set #####################################################################
    G = []
    g = []
    buffer = []
    buffer2 = []
    for i in range(dim_p):
        buffer.append(1)
        buffer2.append(-1)
        g.append(0)
        buffer1 = []
        for j in range(dim_p):
            if (j == i):
                buffer1.append(-1)
            else:
                buffer1.append(0)
        G.append(buffer1)
    G.append(buffer)
    g.append(1)
    G.append(buffer2)
    g.append(-1)

    # define the known constraints
    ak = []
    gk = []

    #    [ak,gk] = Knownconstrs(opt_g)
    dim_known = len(ak)

    # main here #########################################################################################
    eps1 = 1
    eps2 = 1

    if warmstart:
        initt = initialMIP(opt_g, dim_p)
        [opt_b, opt_B, optr, optval] = inverse(xiv, opt_g, demand, dim_p, initt)
        [A2, C2] = FindAC(opt_b, opt_B, dim_p, dim_f)
        A = A2  # warm start from some points
        C = C2
        Cb = rd.rand(dim_f, 5)
    else:
        A = rd.rand(dim_f, dim_p)
        C = rd.rand(5, dim_f, dim_p)
        Cb = rd.rand(dim_f, 5)

    iter_num = 100
    threshold = 0.005
    step = 2
    d2 = len(G)  # number of the constraints

    record_res1 = []
    record_res2 = []
    record_eps = []
    record_A = []
    iter_val = []
    iter_true = []
    buffer1 = np.zeros((datanum, dim_f))
    buffer2 = np.zeros((datanum, dim_p))
    buffer3 = np.zeros((datanum, dim_p))

    record_A.append(A)
    [opt_beta, opt_mu, opt_z, value, opt_b, opt_cb, res1, res2] = invfixAReg2(A, C, opt_g, xiv, eps1, eps2, G, g, ak, gk, demand,
                                                                  fixCb)
    flag_new = 1
    record_res2.append(res2)
    record_res1.append(res1)

    now = time.time()
    for i_n in range(iter_num):
        record_A.append(A)
        val = np.zeros((dim_f, dim_p))
        for i in range(dim_f):
            for j in range(dim_p):
                for k in range(datanum):
                    val[i][j] = (val[i][j] - xiv[k][i] * opt_mu[k][j] + opt_beta[k][i] * opt_z[k][j])

        valc = np.zeros((5, dim_f, dim_p))
        for m in range(5):
            for i in range(dim_f):
                for j in range(dim_p):
                    for k in range(datanum):
                        valc[m][i][j] = (valc[m][i][j] - demand[k][m] * xiv[k][i] * opt_mu[k][j] + demand[k][m] *
                                         opt_beta[k][i] * opt_z[k][j])

        A1 = A - step * val
        C1 = np.zeros((5, dim_f, dim_p))
        for i in range(5):
            C1[i] = C[i] - step * valc[i]

        [opt_beta1, opt_mu1, opt_z1, value1, opt_b1, opt_cb1, res1, res2] = invfixAReg2(A1, C1, opt_g, xiv, eps1, eps2, G, g, ak,
                                                                            gk, demand, fixCb)

        while value1 - value > -step*0.5*sum(val[i][j]**2 for i in range(dim_f) for j in range(dim_p)) and flag_new==0:
            step = step * 0.5
            if step < 0.00001:
                break
            A1 = A - step * val
            for i in range(5):
                C1[i] = C[i] - step * valc[i]
            print('Recalculating the step length!')
            [opt_beta1, opt_mu1, opt_z1, value1, opt_b1, opt_cb1, res1, res2] = invfixAReg2(A1, C1, opt_g, xiv, eps1, eps2, G, g, ak,
                                                                            gk, demand, fixCb)

        if value1 - value <= -step*0.5*sum(val[i][j]**2 for i in range(dim_f) for j in range(dim_p)) or flag_new:
            opt_beta = opt_beta1
            opt_mu = opt_mu1
            opt_z = opt_z1
            opt_b = opt_b1
            opt_cb = opt_cb1
            A = A1
            C = C1
            value = value1
            iter_val.append(value)
            step = 1
            record_res2.append(res2)
            record_res1.append(res1)
            record_eps.append(eps1)
            flag_new = 0
        else:
            break

        if abs(record_res1[-1] - record_res1[-2])<=threshold/eps1 and abs(record_res2[-1]-record_res2[-2])<=threshold/eps2:
            eps1 = eps1*2
            eps2 = eps2*2
            if eps1 > 20:
                break
            flag_new = 1
            threshold = threshold/5
            print(f'Recalculate dual variables with new eps = {eps1}.')
            [opt_beta, opt_mu, opt_z, value, opt_b, opt_cb, res1, res2] = invfixAReg2(A, C, opt_g, xiv, eps1, eps2, G, g, ak,
                                                                            gk, demand, fixCb)
        if (value < 0.000001):
            break
    end = time.time()
    draw_obj(iter_val)
    draw_obj(record_res1)
    draw_obj(record_eps)


    Post_b = opt_b

    # [Post_L,  Post_z, Post_b, Postval] = trueInv(A,C,opt_g,xiv,G,g,ak,gk,demand)

    optx = []
    optx_true = []
    out_obj = 0
    for i in range(test_num):
        [opt_x, optr] = forward2(test_price[i], test_demand[i], Post_b, A, C, opt_cb, G, g, ak, gk)
        [opt_x2, keke] = powersys(test_price[i], test_demand[i], 0)
        optx.append(opt_x)  # optimal solutions recovered in our model
        optx_true.append(opt_x2)  # true optimal solutions

    [residual, true_predict] = cL_p_true(test_price, test_demand, optx, optx_true)
    true_subopt = cL_sub_true(test_price, test_demand, optx, optx_true)
    out_predict = cL_pA(optx_true, test_price, G, g, 5, test_demand, C, A, opt_cb, Post_b)
    out_subopt = cL_sA(optx_true, test_price, G, g, 5, test_demand, C, A, opt_cb, Post_b)
    print(
        f"out_predict={out_predict}, true_predict = {true_predict}, out_subopt={out_subopt}, true_subopt = {true_subopt}.")
    print(f'Running time is {end-now}')
    # out_obj2=Evaluate(A,C,Post_b,optx_true,test_price,G,g,ak,gk,test_demand)
    # vio = violations(optx, test_demand)
    #
    # out_pre= 1/test_num*sum((optx[i][j]-optx_true[i][j])**2 for i in range(test_num) for j in range(dim_f))
    # out_d = 1/test_num*sum(vio)
    # out_pre1= 1/test_num*sum(sum((optx[i][j]-optx_true[i][j])**2 for j in range(dim_f))/sum(optx_true[i][j]**2 for j in range(dim_f))
    # for i in range(test_num))
    # out_d1 = 1/test_num*sum(vio[i]/sum(test_demand[i]) for i in range(test_num))
    # in_obj2=Evaluate(A,C,Post_b,opt_g,xiv,G,g,ak,gk,demand)




























