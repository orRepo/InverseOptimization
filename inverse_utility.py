import gurobipy as gp
from gurobipy import GRB
import numpy.random as rd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def EnergyData(fix, dim_f, datanum, noise, G2, g2, seeds):
    xiv = rd.uniform(0.2, 1, (datanum, dim_f))
    opt_g = np.ones((datanum, dim_f))
    opt_true = np.ones((datanum, dim_f))
    rd.seed(seeds)
    if fix:
        demand = np.ones((datanum, dim_f))
    else:
        demand = np.ones((datanum, dim_f))
        demand[:, 0] = rd.uniform(0.3, 1.5, (datanum,))
        demand[:, 1] = rd.uniform(0.36, 1.8, (datanum,))
        demand[:, 2] = rd.uniform(0.42, 2.1, (datanum,))
        demand[:, 3] = rd.uniform(0.48, 2.4, (datanum,))
        demand[:, 4] = rd.uniform(0.54, 2.7, (datanum,))

    for n in range(datanum):
        [opt_g[n, :], opt_true[n, :]] = powersys(xiv[n, :], demand[n, :], noise, G2, g2)

    return [demand, xiv, np.array(opt_g), np.array(opt_true)]


def powersys(xiv, d, noise, G2, g2):
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n = 5  # num of the arcs

    m = len(xiv)
    g = M1.addMVar( m, lb=0)
    f = M1.addMVar( n, lb=-3.5, ub=3.5)

    M1.setObjective(xiv @ g, GRB.MINIMIZE)

    M1.addConstr(G2 @ g <= g2)
    M1.addConstr(g[0] + f[0] - f[1] >= d[0])
    M1.addConstr(f[1] + f[3] >= d[1])
    M1.addConstr(g[2] - f[0] - f[2] >= d[2])
    M1.addConstr(f[2] + f[4] - f[3] >= d[3])
    M1.addConstr(g[4] - f[4] >= d[4])

    M1.optimize()

    opt_gtrue = g.X
    opt_g = g.X
    if noise:
        for j in range(m):
            opt_g[j] = opt_g[j] + rd.normal(0, 0.1)
            # opt_g[j] = opt_g[j]+rd.laplace(0,0.1/math.sqrt(2))
    return [opt_g, opt_gtrue]

def synthetic_forward(dimension, A , b):
    m = gp.Model("linear")
    m.setParam('TimeLimit', 3600)
    d = dimension

    c = {}
    for i in range(d):
        c[i] = rd.uniform(-1, 1)
    x = {}
    for i in range(d):
        x[i] = m.addVar(lb=-GRB.INFINITY)  # define the variables

    m.setObjective(gp.quicksum(c[j] * x[j] for j in range(d)), GRB.MINIMIZE)  # define the objective

    m.addConstrs(sum(A[i][j]*x[j] for j in range(d)) <= b[i] for i in range(len(b)))
    m.optimize()
    xopt = np.zeros(d)
    for i in range(d):
        xopt[i] = x[i].X

    return (xopt, c)


def synthetic_forward1(dimension, e , b):
    m = gp.Model("linear")
    m.setParam('TimeLimit', 3600)
    d = dimension

    c = {}
    for i in range(d):
        c[i] = rd.uniform(0, 1)
    x = {}
    for i in range(d):
        x[i] = m.addVar(lb=-GRB.INFINITY)  # define the variables
    xabs = {}
    for i in range(d):
        xabs[i] = m.addVar()

    m.setObjective(gp.quicksum(c[j] * x[j] for j in range(d)), GRB.MINIMIZE)  # define the objective

    m.addConstrs(xabs[i] >= x[i] - e[i] for i in range(d))
    m.addConstrs(xabs[i] >= e[i] - x[i] for i in range(d))
    m.addConstr(gp.quicksum(xabs[i] for i in range(d)) <= b)
    m.optimize()
    xopt = np.zeros(d)
    for i in range(d):
        xopt[i] = x[i].X

    return (xopt, c)

def RecoverEnergy(c, demand, b, A, C, Cb, G, g):
    N = len(G)
    m = len(c)
    n = len(G[0])

    M1 = gp.Model("lp1")
    M1.setParam('TimeLimit', 360)

    x={}
    for j in range(m):
        x[j] = M1.addVar(lb= -GRB.INFINITY)
    z={}
    for j in range(n):
        z[j] = M1.addVar(lb= -GRB.INFINITY)

    M1.setObjective(gp.quicksum(c[j]*x[j] for j in range(m)), GRB.MINIMIZE)
    M1.addConstrs(gp.quicksum(G[i][j]*z[j] for j in range(n)) <= g[i] for i in range(N))
    M1.addConstrs(x[i]==gp.quicksum( (A[i][j]+sum(C[k][i][j]*demand[k] for k in range(len(demand))))*z[j]for j in range(n))
                  + b[i] + gp.quicksum(Cb[i][k]*demand[k] for k in range(len(demand))) for i in range(m))
    #M1.addConstrs(gp.quicksum(ak[j][i]*x[i] for i in range(m))<=gk[j] for j in range(len(gk)))

    M1.optimize()
    optx = np.zeros(m)
    for j in range(m):
        optx[j] = x[j].X

    return optx

def forward2(c, b, a, G, g):  # recovered forward problem
    N = len(G)
    m = 2
    M1 = gp.Model("lp1")
    M1.setParam('TimeLimit', 360)

    x = {}
    for j in range(m):
        x[j] = M1.addVar(lb=-GRB.INFINITY)
    z = {}
    for j in range(m):
        z[j] = M1.addVar(lb=-GRB.INFINITY)

    M1.setObjective(gp.quicksum(c[j] * x[j] for j in range(m)), GRB.MINIMIZE)
    M1.addConstrs(gp.quicksum(G[i][j] * z[j] for j in range(m)) <= g[i] for i in range(N))
    M1.addConstrs(x[i] == a * z[i] + b[i] for i in range(m))

    M1.optimize()
    optx = np.zeros(m)
    optz = np.zeros(m)
    for j in range(m):
        optx[j] = x[j].X
    for j in range(m):
        optz[j] = z[j].X
    return [optx, optz]


def forward2A(c, b, A, G, g):
    N = len(G)
    m = len(b)
    n = len(G[0])

    M1 = gp.Model("lp1")
    M1.setParam('TimeLimit', 3600)

    x = {}
    for j in range(m):
        x[j] = M1.addVar(lb=-GRB.INFINITY)
    z = {}
    for j in range(n):
        z[j] = M1.addVar(lb=-GRB.INFINITY)

    M1.setObjective(gp.quicksum(c[j] * x[j] for j in range(m)), GRB.MINIMIZE)
    M1.addConstrs(gp.quicksum(G[i][j] * z[j] for j in range(n)) <= g[i] for i in range(N))
    M1.addConstrs(x[i] == gp.quicksum(A[i][j] * z[j] for j in range(n)) + b[i] for i in range(m))
    M1.optimize()
    optx = np.zeros(m)
    for j in range(m):
        optx[j] = x[j].X
    return optx


def L_p_true(c, x, xtrue, H, h): # Loss function: true predictability loss, H,h true feasible region
    # x: recovered optimal; x_true: true optimal
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n = len(x)  # number of data points
    m = len(x[0])  # dimension of the decision variable
    gamma = {}
    xx = {}
    for j in range(n):
        gamma[j] = {}
        xx[j]={}
        for i in range(m):
            gamma[j][i] = M1.addVar(lb=-GRB.INFINITY)
            xx[j][i] = M1.addVar(lb=-GRB.INFINITY)
    M1.setObjective(gp.quicksum(gamma[j][i]*gamma[j][i] for j in range(n) for i in range(m)), GRB.MINIMIZE)

    M1.addConstrs(x[i][j]+gamma[i][j]==xx[i][j] for i in range(n) for j in range(m))
    M1.addConstrs(sum(H[i][j]*xx[k][j] for j in range(m)) <= h[i] for i in range(len(h)) for k in range(n))

    #M1.addConstrs(xx[i][0] + xx[i][1] <= 2 + math.sqrt(2) for i in range(n))  # define the constraints
    #M1.addConstrs(-xx[i][0] + xx[i][1] <= math.sqrt(2) for i in range(n))
    #M1.addConstrs(xx[i][0] - xx[i][1] <= math.sqrt(2) for i in range(n))
    #M1.addConstrs(-xx[i][0] - xx[i][1] <= -2 + math.sqrt(2) for i in range(n))
    M1.addConstrs(sum(c[i][j]*xx[i][j] for j in range(m)) - sum(c[i][j]*xtrue[i][j] for j in range(m)) <=0 for i in range(n))

    M1.optimize()

    residual = {}
    for i in range(n):
        residual[i] = np.zeros(m)
        for j in range(m):
            residual[i][j] = gamma[i][j].X

    return [residual, M1.objVal/n]

def L_p_true1(c, x, xtrue, e, h): # Loss function: true predictability loss, H,h true feasible region
    # x: recovered optimal; x_true: true optimal
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n = len(x)  # number of data points
    m = len(x[0])  # dimension of the decision variable
    gamma = {}
    xxabs = {}
    for j in range(n):
        gamma[j] = {}
        xxabs[j] = {}
        for i in range(m):
            gamma[j][i] = M1.addVar(lb=-GRB.INFINITY)
            xxabs[j][i] = M1.addVar()

    M1.setObjective(gp.quicksum(gamma[j][i]**2 for j in range(n) for i in range(m)), GRB.MINIMIZE)

    #M1.addConstrs(x[i][j]+gamma[i][j]==xx[i][j] for i in range(n) for j in range(m))
    M1.addConstrs(xxabs[i][j] >= x[i][j]+gamma[i][j] - e[j] for i in range(n) for j in range(m))
    M1.addConstrs(xxabs[i][j] >= e[j] - x[i][j]-gamma[i][j] for i in range(n) for j in range(m))
    M1.addConstrs(sum(xxabs[i][j] for j in range(m)) <= h for i in range(n))

    M1.addConstrs(sum(c[i][j]*(x[i][j]+gamma[i][j]) for j in range(m)) -
                  sum(c[i][j]*xtrue[i][j] for j in range(m)) <=0 for i in range(n))

    M1.optimize()

    residual = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            residual[i][j] = gamma[i][j].X

    return [residual, M1.objVal/n]


def L_sub_true(c, x, xtrue, H, h): # true sub-optimality loss
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n = len(x)  # number of data points
    m = len(x[0])  # dimension of the decision variable
    rr = len(H) # number of constraints
    gammaf = {}
    gammao = {}
    for j in range(n):
        gammaf[j] = {}
        gammao[j]=M1.addVar()
        for i in range(rr):
            gammaf[j][i] = M1.addVar()
    M1.setObjective(gp.quicksum(gammaf[j][i]*gammaf[j][i] for j in range(n) for i in range(rr))
                    +gp.quicksum(gammao[j]*gammao[j] for j in range(n)), GRB.MINIMIZE)

    M1.addConstrs(float(sum(H[i][j]*x[k][j] for j in range(m))) <= h[i] + gammaf[k][i] for i in range(rr) for k in range(n))
    M1.addConstrs(float(sum(c[i][j]*x[i][j] for j in range(m)) - sum(c[i][j]*xtrue[i][j] for j in range(m))) <= gammao[i] for i in range(n))

    M1.optimize()
    # residual = {}
    # for i in range(n):
    #     residual[i] = np.zeros(m)
    #     for j in range(m):
    #         residual[i][j] = gamma[i][j].X
    return M1.objVal/n

def L_sub_true1(c, x, xtrue, e, h): # true sub-optimality loss
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n = len(x)  # number of data points
    m = len(x[0])  # dimension of the decision variable
    gammaf = {}
    gammao = {}
    xabs={}
    for j in range(n):
        gammaf[j] = M1.addVar()
        gammao[j]=M1.addVar()
        xabs[j] = {}
        for i in range(m):
            xabs[j][i] = M1.addVar()
    M1.setObjective(gp.quicksum(gammaf[j]*gammaf[j] for j in range(n))
                    +gp.quicksum(gammao[j]*gammao[j] for j in range(n)), GRB.MINIMIZE)

    M1.addConstrs(xabs[i][j] >= x[i][j] -e[j] for i in range(n) for j in range(m))
    M1.addConstrs(xabs[i][j] >= -x[i][j]+e[j] for i in range(n) for j in range(m))
    M1.addConstrs(gp.quicksum(xabs[i][j] for j in range(m)) <= h + gammaf[i] for i in range(n))
    M1.addConstrs(float(sum(c[i][j]*x[i][j] for j in range(m)) - sum(c[i][j]*xtrue[i][j] for j in range(m))) <= gammao[i] for i in range(n))

    M1.optimize()
    return M1.objVal/n


def L_p(c, x, G, g, a, b):  # out-of-sample predictability loss;
    m = gp.Model("socp")
    m.setParam('TimeLimit', 3600)
    d = len(x[0])
    N = len(x)
    K=len(g)
    gammao = {}
    for i in range(N):
        gammao[i] = {}
        for j in range(d):
            gammao[i][j] = m.addVar(lb=-GRB.INFINITY)
    v = {}
    for n in range(N):
        v[n] = {}
        for i in range(d):
            v[n][i] = m.addVar(lb=-GRB.INFINITY)
    lamb = {}
    for n in range(N):
        lamb[n] = {}
        for k in range(K):
            lamb[n][k] = m.addVar()
    m.setObjective( gp.quicksum(gp.quicksum(gammao[n][i] ** 2 for i in range(d)) for n in range(N)), GRB.MINIMIZE)  # define the objective

    m.addConstrs(x[n][i] + gammao[n][i] == v[n][i] + b[i] for i in range(d) for n in range(N))
    m.addConstrs(gp.quicksum(G[k][i] * v[n][i] for i in range(d)) <= a * g[k] for k in range(K) for n in range(N))
    #m.addConstrs(gammao[i][j] == gammaf[i][j] for i in range(N) for j in range(d))
    m.addConstrs(gp.quicksum(c[n][i] * (x[n][i] + gammao[n][i]) for i in range(d)) - gp.quicksum(
        c[n][i] * b[i] for i in range(d)) + gp.quicksum(lamb[n][k] * g[k] for k in range(K)) <=0 for n in range(N))
    m.addConstrs(a * c[n][i] + gp.quicksum(lamb[n][k] * G[k][i] for k in range(K)) == 0 for i in range(d) for n in range(N))
    #m.addConstrs(lamb[n][k] >= 0 for n in range(N) for k in range(K))

    m.optimize()

    return m.objVal/N

def L_pA(c, x, G, g, A, b):  # out-of-sample predictability loss;
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G)  # num of constraints
    n = len(x)  # data num

    m = gp.Model("main")
    m.setParam('TimeLimit', 3600)

    gammao = {}
    for i in range(n):
        gammao[i] = {}
        for j in range(d1):
            gammao[i][j] = m.addVar(lb=-GRB.INFINITY)

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

    m.setObjective(1 / n * gp.quicksum( gammao[i][j] ** 2 for i in range(n) for j in range(d1)),
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))

    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(A[j][k] * z[i][k] for k in range(d)) - b[j] + gammao[i][j] == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(gp.quicksum(c[i][j] * A[j][k] for j in range(d1))
                                   + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] + gammao[i][j] - b[j]) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= 0 for i in range(n))

    m.optimize()

    return m.objVal



def L_sub2(c, x, a, G, g, b):
    m = gp.Model("socp")
    m.setParam('TimeLimit', 3600)
    N = len(x)
    K = len(g)
    d = 2

    gammao = {}
    for i in range(N):
        gammao[i] = m.addVar()

    gammaf = {}
    for i in range(N):
        gammaf[i] = {}
        for j in range(d):
            gammaf[i][j] = m.addVar(lb=-GRB.INFINITY)

    absgammaf = {}
    for i in range(N):
        absgammaf[i] = {}
        for j in range(d):
            absgammaf[i][j] = m.addVar(lb=-GRB.INFINITY)

    v = {}
    for n in range(N):
        v[n] = {}
        for i in range(d):
            v[n][i] = m.addVar(lb=-GRB.INFINITY)

    lamb = {}
    for n in range(N):
        lamb[n] = {}
        for k in range(K):
            lamb[n][k] = m.addVar()

    m.setObjective(gp.quicksum(gammaf[i][j] ** 2 for i in range(N) for j in range(d))
                   + gp.quicksum(gammao[i] ** 2 for i in range(N)), GRB.MINIMIZE)  # define the objective

    m.addConstrs(absgammaf[i][j] >= gammaf[i][j] for i in range(N) for j in range(d))
    m.addConstrs(absgammaf[i][j] >= -gammaf[i][j] for i in range(N) for j in range(d))

    m.addConstrs(x[n][i] + gammaf[n][i] == v[n][i] + b[i] for i in range(d) for n in range(N))

    m.addConstrs(gp.quicksum(G[k][i] * v[n][i] for i in range(d)) <= a * g[k] for k in range(K) for n in range(N))

    m.addConstrs(gammao[n] >= gp.quicksum(c[n][i] * x[n][i] for i in range(d)) - gp.quicksum(c[n][i] * b[i]
                                                                                             for i in
                                                                                             range(d)) + gp.quicksum(
        lamb[n][k] * g[k] for k in range(K)) for n in range(N))

    m.addConstrs(
        a * c[n][i] + gp.quicksum(lamb[n][k] * G[k][i] for k in range(K)) == 0 for i in range(d) for n in range(N))

    m.optimize()
    return m.objVal/N

def L_subA(c, x, G, g, A, b): # out-of-sample suboptimality loss
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G)  # num of constraints
    n = len(x)  # data num

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

    m.setObjective(gp.quicksum(gammaf[i][j] ** 2 for i in range(n) for j in range(d1))
                   + gp.quicksum(gammao[i]**2 for i in range(n)), GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(A[j][k] * z[i][k] for k in range(d)) - b[j] - gammaf[i][j] == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(gp.quicksum(c[i][j] * A[j][k] for j in range(d1))
                                   + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] - b[j]) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= gammao[i] for i in range(n))

    m.optimize()

    return m.objVal/n

def L_sub(c, x, G, g, a, b): # out-of-sample suboptimality loss
    m = gp.Model("socp")
    m.setParam('TimeLimit', 3600)
    N = len(x)
    d = len(x[0])
    K = len(g)
    gammao = {}
    for i in range(N):
        gammao[i] = m.addVar()

    gammaf = {}
    for i in range(N):
        gammaf[i] = {}
        for j in range(d):
            gammaf[i][j] = m.addVar(lb=-GRB.INFINITY)

    v = {}
    for n in range(N):
        v[n] = {}
        for i in range(d):
            v[n][i] = m.addVar(lb=-GRB.INFINITY)

    lamb = {}
    for n in range(N):
        lamb[n] = {}
        for k in range(K):
            lamb[n][k] = m.addVar()

    m.setObjective(gp.quicksum(gammaf[i][j]**2 for i in range(N) for j in range(d))
                   + gp.quicksum(gammao[i]**2 for i in range(N)), GRB.MINIMIZE)  # define the objective

    m.addConstrs(x[n][i] + gammaf[n][i] == v[n][i] + b[i] for i in range(d) for n in range(N))
    m.addConstrs(gp.quicksum(G[k][i] * v[n][i] for i in range(d)) <= a * g[k] for k in range(K) for n in range(N))
    m.addConstrs(gp.quicksum(c[n][i] * x[n][i] for i in range(d)) - gp.quicksum(c[n][i] * b[i] for i in range(d))
                 + gp.quicksum( lamb[n][k] * g[k] for k in range(K)) <= gammao[n] for n in range(N))

    m.addConstrs(
        a * c[n][i] + gp.quicksum(lamb[n][k] * G[k][i] for k in range(K)) == 0 for i in range(d) for n in range(N))

    m.optimize()

    return m.objVal/N


def invfixAP(A, x, c, G, g, eps1, eps2):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G) # num of constraints
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

    gammanew = {}
    for i in range(n):
        gammanew[i] = {}
        for j in range(d):
            gammanew[i][j] = m.addVar(lb=-GRB.INFINITY)

    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)

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

    m.setObjective(1/n*gp.quicksum(eps1*(gammaf[i][j])**2 + gammao[i][j]**2 for i in range(n) for j in range(d1))
                   + eps2/n*gp.quicksum(gammanew[i][j]**2 for i in range(n) for j in range(d)),
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    #m.addConstrs(gammaf[i][j] == gammao[i][j] for i in range(n) for j in range(d1))
    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(A[j][k] * z[i][k] for k in range(d)) - b[j] + gammaf[i][j] + gammao[i][j] == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(gp.quicksum(c[i][j] * A[j][k] for j in range(d1)) + gammanew[i][k]
                                   + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] + gammao[i][j] - b[j]) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= 0 for i in range(n))

    m.optimize()

    opt_beta = np.zeros((n, d1))
    opt_mu = np.zeros((n, d))
    opt_z = np.zeros((n, d))
    optval = m.objVal
    opt_b = np.zeros(d1)
    for i in range(n):
        for j in range(d1):
            opt_beta[i][j] = beta[i][j].Pi
        for k in range(d):
            opt_z[i][k] = z[i][k].X
            opt_mu[i][k] = mu[i][k].Pi
    for i in range(d1):
        opt_b[i] = b[i].X

    res1 = sum(gammanew[i][j].X*gammanew[i][j].X for i in range(n) for j in range(d))
    res2 = sum(gammaf[i][j].X**2 for i in range(n) for j in range(d1))

    return [opt_beta, opt_mu, opt_z, optval, opt_b, res1, res2]

def invfixAP_orig(A, x, c, G, g):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G) # num of constraints
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

    # gammanew = {}
    # for i in range(n):
    #     gammanew[i] = {}
    #     for j in range(d):
    #         gammanew[i][j] = m.addVar(lb=-GRB.INFINITY)

    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)

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

    m.setObjective(1/n*gp.quicksum(gammaf[i][j]**2 + gammao[i][j]**2 for i in range(n) for j in range(d1)),
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    m.addConstrs(gammaf[i][j] == gammao[i][j] for i in range(n) for j in range(d1))
    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(A[j][k] * z[i][k] for k in range(d)) - b[j] + gammaf[i][j] == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(gp.quicksum(c[i][j] * A[j][k] for j in range(d1))
                                   + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] + gammao[i][j] - b[j]) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= 0 for i in range(n))

    m.optimize()

    opt_beta = np.zeros((n, d1))
    opt_mu = np.zeros((n, d))
    opt_z = np.zeros((n, d))
    optval = m.objVal
    opt_b = np.zeros(d1)
    for i in range(n):
        for j in range(d1):
            opt_beta[i][j] = beta[i][j].Pi
        for k in range(d):
            opt_z[i][k] = z[i][k].X
            opt_mu[i][k] = mu[i][k].Pi
    for i in range(d1):
        opt_b[i] = b[i].X

    return [opt_beta, opt_mu, opt_z, optval, opt_b]

def invPSim(x, c, G, g):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G) # num of constraints
    n = len(x)  # data num

    m = gp.Model("main")
    m.setParam('TimeLimit', 900)

    A={}
    for i in range(d1):
        A[i]={}
        for j in range(d):
            A[i][j]=m.addVar(lb=-GRB.INFINITY)

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

    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)

    z = {}
    for i in range(n):
        z[i] = {}
        for j in range(d):
            z[i][j] = m.addVar(vtype=GRB.BINARY)

    lamb = {}
    for i in range(n):
        lamb[i] = {}
        for j in range(d2):
            lamb[i][j] = m.addVar()

    m.setObjective(1 / n * gp.quicksum(gammaf[i][j]**2 + gammao[i][j]**2 for i in range(n) for j in range(d1)),
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    m.addConstrs(gammaf[i][j] == gammao[i][j] for i in range(n) for j in range(d1))
    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(A[j][k] * z[i][k] for k in range(d)) - b[j] + gammaf[i][j] == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(gp.quicksum(c[i][j] * A[j][k] for j in range(d1))
                                   + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] + gammao[i][j] - b[j]) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= 0 for i in range(n))

    m.optimize()

    opt_A = np.zeros((d1, d))
    opt_b = np.zeros(d1)
    for i in range(d1):
        for k in range(d):
            opt_A[i][k] = A[i][k].X
    for i in range(d1):
        opt_b[i] = b[i].X

    return [opt_A, opt_b]


def invPSimC(x, c, G, g, dimC, delta, A2, C2):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G) # num of constraints
    n = len(x)  # data num
    nL = dimC
    m = gp.Model("main")
    m.setParam('TimeLimit', 2000)

    A={}
    for i in range(d1):
        A[i]={}
        for j in range(d):
            A[i][j]=m.addVar(lb=-GRB.INFINITY)
    C = {}
    for i in range(dimC):
        C[i] = {}
        for j in range(d1):
            C[i][j] = {}
            for k in range(d):
                C[i][j][k] = m.addVar(lb=-GRB.INFINITY)
    C_b = {}
    for i in range(d1):
        C_b[i]={}
        for j in range(dimC):
            C_b[i][j] = m.addVar(lb=-GRB.INFINITY)
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
    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)
    z = {}
    for i in range(n):
        z[i] = {}
        for j in range(d):
            z[i][j] = m.addVar(vtype=GRB.BINARY)

    lamb = {}
    for i in range(n):
        lamb[i] = {}
        for j in range(d2):
            lamb[i][j] = m.addVar()

    m.setObjective(1 / n * gp.quicksum(gammaf[i][j]**2 + gammao[i][j]**2 for i in range(n) for j in range(d1)),
                   GRB.MINIMIZE)

    #m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    m.addConstrs(gp.quicksum(z[i][j] for j in range(d)) == 1 for i in range(n))
    for i in range(n):
        m.addSOS(GRB.SOS_TYPE1, z[i])

    m.addConstrs(gammaf[i][j] == gammao[i][j] for i in range(n) for j in range(d1))

    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum((A[j][k]+gp.quicksum(C[r][j][k]*delta[i][r] for r in range(nL)))*z[i][k] for k in range(d))
                - b[j] - gp.quicksum(C_b[j][k]*delta[i][k] for k in range(nL)) + gammaf[i][j] == 0)
    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(gp.quicksum(c[i][j] * (A[j][k]+gp.quicksum(C[r][j][k]*delta[i][r] for r in range(nL)))
                                    for j in range(d1)) + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] + gammao[i][j] - b[j]
                - gp.quicksum(C_b[j][k]*delta[i][k] for k in range(nL))) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= 0 for i in range(n))


    for i in range(d1):
        for j in range(d):
            A[i][j].start=A2[i][j]
    for i in range(dimC):
        for j in range(d1):
            for k in range(d):
                C[i][j][k].start = C2[i][j][k]

    m.optimize()

    opt_A = np.zeros((d1, d))
    opt_C = np.zeros((nL,d1,d))
    opt_Cb = np.zeros((d1,nL))
    opt_b = np.zeros(d1)
    for i in range(d1):
        for k in range(d):
            opt_A[i][k] = A[i][k].X
            for j in range(nL):
                opt_C[j][i][k] = C[j][i][k].X
    for i in range(d1):
        opt_b[i] = b[i].X
        for j in range(nL):
            opt_Cb[i][j] = C_b[i][j].X

    return [opt_A, opt_C, opt_Cb, opt_b]

def invfixAS(A, x, c, G, g, eps1):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G) # num of constraints
    n = len(x)  # data num

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

    gammanew = {}
    for i in range(n):
        gammanew[i] = {}
        for j in range(d):
            gammanew[i][j] = m.addVar(lb=-GRB.INFINITY)

    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)

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

    m.setObjective(1/n * gp.quicksum(gammaf[i][j] ** 2 for i in range(n) for j in range(d1))
                   + 1/n*gp.quicksum(gammao[i] ** 2 for i in range(n)) + eps1/n*gp.quicksum(gammanew[i][j]**2
                     for i in range(n) for j in range(d)),
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    #m.addConstrs(gammaf[i][j] == gammao[i][j] for i in range(n) for j in range(d1))
    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(A[j][k] * z[i][k] for k in range(d)) - b[j] + gammaf[i][j] == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(gp.quicksum(c[i][j] * A[j][k] for j in range(d1)) + gammanew[i][k]
                                   + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)


    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] - b[j]) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= gammao[i] for i in range(n))

    m.optimize()

    opt_beta = np.zeros((n, d1))
    opt_mu = np.zeros((n, d))
    opt_z = np.zeros((n, d))
    optval = m.objVal
    opt_b = np.zeros(d1)
    for i in range(n):
        for j in range(d1):
            opt_beta[i][j] = beta[i][j].Pi
        for k in range(d):
            opt_z[i][k] = z[i][k].X
            opt_mu[i][k] = mu[i][k].Pi
    for i in range(d1):
        opt_b[i] = b[i].X

    res1 = sum(gammanew[i][j].X * gammanew[i][j].X for i in range(n) for j in range(d))

    return [opt_beta, opt_mu, opt_z, optval, opt_b, res1, 0]

def invfixAS_orig(A, x, c, G, g):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G) # num of constraints
    n = len(x)  # data num

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

    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)

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

    m.setObjective(1/n * gp.quicksum(gammaf[i][j] ** 2 for i in range(n) for j in range(d1))
                   + 1/n*gp.quicksum(gammao[i] ** 2 for i in range(n)),
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    #m.addConstrs(gammaf[i][j] == gammao[i][j] for i in range(n) for j in range(d1))
    beta = {}
    for i in range(n):
        beta[i] = {}
        for j in range(d1):
            beta[i][j] = m.addConstr(
                x[i][j] - gp.quicksum(A[j][k] * z[i][k] for k in range(d)) - b[j] + gammaf[i][j] == 0)

    mu = {}
    for i in range(n):
        mu[i] = {}
        for k in range(d):
            mu[i][k] = m.addConstr(gp.quicksum(c[i][j] * A[j][k] for j in range(d1))
                                   + gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2)) == 0)

    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] - b[j]) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= gammao[i] for i in range(n))

    m.optimize()

    opt_beta = np.zeros((n, d1))
    opt_mu = np.zeros((n, d))
    opt_z = np.zeros((n, d))
    optval = m.objVal
    opt_b = np.zeros(d1)
    for i in range(n):
        for j in range(d1):
            opt_beta[i][j] = beta[i][j].Pi
        for k in range(d):
            opt_z[i][k] = z[i][k].X
            opt_mu[i][k] = mu[i][k].Pi
    for i in range(d1):
        opt_b[i] = b[i].X

    return [opt_beta, opt_mu, opt_z, optval, opt_b]

def invfixAPR(A, x, c, G, g, eps1, eps2):
    d = len(G[0])  # dimension of primitive sets
    d1 = len(x[0])  # dimension of the forward problem
    d2 = len(G)
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
    b = {}
    for i in range(d1):
        b[i] = m.addVar(lb=-GRB.INFINITY)
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

    # obj = sum(gamma)+ eps2*(norm(b)^2 + norm(lamb,'fro')^2 + norm(z,'fro')^2)
    # + 1/eps1*norm(optx-A*z'-b*I,'fro')^2 + 1/eps1*norm(c'*A-lamb*G,'fro')^2;
    L4 = 1 / eps1 * gp.quicksum((x[i][j] - gp.quicksum(A[j][k] * z[i][k] for k in range(d)) - b[j] +
                                 gammaf[i][j]) ** 2 for i in range(n) for j in range(d1))
    L5 = 1 / eps2 * gp.quicksum((gp.quicksum(c[i][j] * A[j][k] for j in range(d1)) +
                                 gp.quicksum(lamb[i][j] * G[j][k] for j in range(d2))) ** 2 for i in range(n) for k in
                                range(d))

    m.setObjective(1 / n * gp.quicksum(
        gammaf[i][j] ** 2 + gammao[i][j] ** 2 for i in range(n) for j in range(d1)) + L4 + L5,
                   GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(G[i][j] * z[k][j] for j in range(d)) - g[i] <= 0 for i in range(d2) for k in range(n))
    #m.addConstrs(gammaf[i][j] == gammao[i][j] for i in range(n) for j in range(d1))
    m.addConstrs(gp.quicksum(c[i][j] * (x[i][j] + gammao[i][j] - b[j]) for j in range(d1)) +
                 gp.quicksum(lamb[i][j] * g[j] for j in range(d2)) <= 0 for i in range(n))

    m.optimize()

    opt_L = np.zeros((n, d2))
    opt_b = np.zeros(d1)
    opt_gammaf = np.zeros((n, d))
    opt_z = np.zeros((n, d))
    optval = m.objVal
    for i in range(n):
        for j in range(d2):
            opt_L[i][j] = lamb[i][j].X
        for k in range(d):
            opt_z[i][k] = z[i][k].X
        for k in range(d1):
            opt_gammaf[i][k] = gammaf[i][k].X
    for i in range(d1):
        opt_b[i] = b[i].X
    return [opt_L, opt_b, opt_z, opt_gammaf, optval]


def inverse_gradient(A, iter_num, G, g, dim_f, dim_p, x, c, type, eps1, eps2):
    record_A = []
    record_res1 = []
    record_res2 = []
    record_eps = []
    datanum = len(x)
    step = 1
    iter_val = []
    threshold = 0.01/(6**(math.log(eps1, 2)+1))

    record_A.append(A)
    record_eps.append(eps1)
    if type == 'pre':
        print("Solve the starting problem")
        [opt_beta, opt_mu, opt_z, value, opt_b, res1, res2] = invfixAP(A, x, c, G, g, eps1, eps2)
    elif type == 'sub':
        [opt_beta, opt_mu, opt_z, value, opt_b, res1, res2] = invfixAS(A, x, c, G, g, eps1)
    else:
        print("Unknown loss functions!")
        return 0
    record_res2.append(res2)
    record_res1.append(res1)
    flag_new = 1
    for i_n in range(iter_num):
        record_A.append(A)
        val = np.zeros((dim_f, dim_p))
        for i in range(dim_f):
            for j in range(dim_p):
                for k in range(datanum):
                    val[i][j] = (val[i][j] + c[k][i] * opt_mu[k][j] - opt_beta[k][i] * opt_z[k][j])

        A1 = A + step * val
        if type == 'pre':
            print('start a new interation!')
            [opt_beta1, opt_mu1, opt_z1, value1, opt_b1, res1, res2] = invfixAP(A1, x, c, G, g, eps1, eps2)
        else:
            [opt_beta1, opt_mu1, opt_z1, value1, opt_b1, res1, res2] = invfixAS(A1, x, c, G, g, eps1)

        while value1 - value > -step*0.5*sum(val[i][j]**2 for i in range(dim_f) for j in range(dim_p)) and flag_new==0:
            step = step * 0.5
            if step < 0.00001:
                break
            A1 = A + step * val
            if type == 'pre':
                print('Recalculating the step length!')
                [opt_beta1, opt_mu1, opt_z1, value1, opt_b1, res1, res2] = invfixAP(A1, x, c, G, g, eps1, eps2)
            else:
                [opt_beta1, opt_mu1, opt_z1, value1, opt_b1, res1, res2] = invfixAS(A1, x, c, G, g, eps1)
            print('Recalculating step length finished!')

        if value1 - value <= -step*0.5*sum(val[i][j]**2 for i in range(dim_f) for j in range(dim_p)) or flag_new:
            opt_beta = opt_beta1
            opt_mu = opt_mu1
            opt_z = opt_z1
            opt_b = opt_b1
            A = A1
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
            if eps1 > 100:
                break
            flag_new = 1
            threshold = threshold/6
            print(f'Recalculate dual variables with new eps = {eps1}.')
            if type == 'pre':
                [opt_beta, opt_mu, opt_z, value, opt_b, res1, res2] = invfixAP(A, x, c, G, g, eps1, eps2)
            else:
                [opt_beta, opt_mu, opt_z, value, opt_b, res1, res2] = invfixAS(A, x, c, G, g, eps1)

        if (value < 0.000001):
            break

    return [A, opt_b, iter_val, record_res1, record_res2, record_eps, eps1]

def inverse_gradient_reg(A, iter_num, G, g, dim_f, dim_p, x, c, eps1, eps2):
    record_A = []
    datanum = len(x)
    d2 = len(G)
    step = 0.004
    iter_val = []

    buffer1 = np.zeros((datanum, dim_f))
    buffer2 = np.zeros((datanum, dim_p))
    buffer3 = np.zeros((datanum, dim_p))
    record_A.append(A)
    [opt_l, opt_b, opt_z, opt_gf, value] = invfixAPR(A, x, c, G, g, eps1, eps2)
    for i_n in range(iter_num):
        record_A.append(A)
        # \frac{1}{\epsilon_1}\sum_{n}2(A\z_n+\b-\x_n)\z_n^T+\frac{1}{\epsilon_1}\sum_{n}2\c(\s_n)*(\c^T(\s_n)A-\blambda^T_nG)
        for i in range(datanum):
            for j in range(dim_f):
                buffer1[i][j] = sum(A[j][k] * opt_z[i][k] for k in range(dim_p))
        for i in range(datanum):
            for j in range(dim_p):
                buffer2[i][j] = sum(c[i][k] * A[k][j] for k in range(dim_f))
                buffer3[i][j] = sum(opt_l[i][k] * G[k][j] for k in range(d2))
        # buffer1[i][j] = gp.quicksum(A[j][k]*opt_z[i][k] for k in range(dim_p)) for j in range(dim_f) for i in range(datanum) #A*z_n
        # buffer2[i][j] = gp.quicksum(c[i][k]*A[k][j] for k in range(dim_f)) for i in range(datanum) for j in range(dim_p)
        # buffer3[i][j] = gp.quicksum(lamb[i][k]*G[k][j] for k in range(d2)) for i in range(datanum) for j in range(dim_p)
        val = np.zeros((dim_f, dim_p))
        for i in range(dim_f):
            for j in range(dim_p):
                for k in range(datanum):
                    val[i][j] = val[i][j] = (
                                val[i][j] + 2./eps1 * (buffer1[k][i] + opt_b[i] - opt_gf[k][i] - x[k][i]) * opt_z[k][j]
                                + 2./eps2 * c[k][i] * (buffer2[k][j] + buffer3[k][j]))

        A1 = A - step * val
        [opt_l1, opt_b1, opt_z1, opt_gf1, value1] = invfixAPR(A1, x, c, G, g, eps1, eps2)

        if (value < value1+0.00001):
            step = step * 0.8
        else:
            opt_l = opt_l1
            opt_b = opt_b1
            opt_z = opt_z1
            A = A1
            value = value1
            opt_gf = opt_gf1
        iter_val.append(value)
        if (step < 10 ** (-8)):
            break
    return [A, opt_b, iter_val]

def inverse_gradient_regular(A, iter_num, G, g, dim_f, dim_p, x, c, type):
    record_A = []
    datanum = len(x)
    d2 = len(G)
    step = 0.1
    iter_val = []

    record_A.append(A)
    if type == 'pre':
        # print('Recalculating the step length!')
        [opt_beta, opt_mu, opt_z, value, opt_b] = invfixAP_orig(A, x, c, G, g)
    else:
        [opt_beta, opt_mu, opt_z, value, opt_b] = invfixAS_orig(A, x, c, G, g)

    for i_n in range(iter_num):
        record_A.append(A)
        val = np.zeros((dim_f, dim_p))
        for i in range(dim_f):
            for j in range(dim_p):
                for k in range(datanum):
                    val[i][j] = (val[i][j] + c[k][i] * opt_mu[k][j] - opt_beta[k][i] * opt_z[k][j])

        A1 = A + step * val
        if type == 'pre':
            #[opt_beta1, opt_mu1, opt_z1, value1, opt_b1, res1, res2]
            [opt_beta1, opt_mu1, opt_z1, value1, opt_b1] = invfixAP_orig(A1, x, c, G, g)
        else:
            [opt_beta1, opt_mu1, opt_z1, value1, opt_b1] = invfixAS_orig(A1, x, c, G, g)

        while value1 - value > -step * 0.5 * sum(
                val[i][j] ** 2 for i in range(dim_f) for j in range(dim_p)):
            step = step * 0.5
            if step < 0.00001:
                break
            A1 = A + step * val
            if type == 'pre':
                print('Recalculating the step length!')
                [opt_beta1, opt_mu1, opt_z1, value1, opt_b1] = invfixAP_orig(A1, x, c, G, g)
            else:
                [opt_beta1, opt_mu1, opt_z1, value1, opt_b1] = invfixAS_orig(A1, x, c, G, g)
            print('Recalculating step length finished!')

        if value1 - value <= -step * 0.5 * sum(val[i][j] ** 2 for i in range(dim_f) for j in range(dim_p)):
            opt_beta = opt_beta1
            opt_mu = opt_mu1
            opt_z = opt_z1
            opt_b = opt_b1
            A = A1
            value = value1
            iter_val.append(value)
            step = 0.1
        else:
            break
    return [A, opt_b, iter_val]

def genSimplex(dim_p):
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
    return [G,g]

def draw_obj(iter_val):
    plt.plot( iter_val, 'b-')
    # for ren in range(10):
    #     plt.scatter(ren*3, iter_val[ren*3])
    plt.show()

def gendata(datanum, d, seeds, noise, A, b):
    signal=np.zeros((datanum,d))
    observed=np.zeros((datanum,d))
    true=np.zeros((datanum,d))
    rd.seed(seeds)
    for j in range(datanum):
        (xopt,c) = synthetic_forward(d, A, b)
        for i in range(d):
            signal[j][i] = c[i]
        for i in range(d):
            if(noise>=3):
                observed[j][i] = xopt[i] + np.random.normal(0, 0.5)  #add a normal noise
            elif(noise>=2):
                observed[j][i] = xopt[i] + np.random.uniform(-1, 1)  #add a uniform noise
            elif(noise>=1):
                observed[j][i] = xopt[i] + signal[j][i]/sum(signal[j][i]*signal[j][i] for i in range(d))
            else:
                observed[j][i] = xopt[i]
            true[j][i] = xopt[i]
    return [signal, observed, true]

def gendata1(datanum, d, seeds, noise, A, b):
    signal=np.zeros((datanum,d))
    observed=np.zeros((datanum,d))
    true=np.zeros((datanum,d))
    rd.seed(seeds)
    for j in range(datanum):
        (xopt,c) = synthetic_forward1(d, A, b)
        for i in range(d):
            signal[j][i] = c[i]
        for i in range(d):
            if(noise>=3):
                observed[j][i] = xopt[i] + np.random.normal(0, 0.2)  #add a normal noise
            elif(noise>=2):
                observed[j][i] = xopt[i] + np.random.uniform(-1, 1)  #add a uniform noise
            elif(noise>=1):
                observed[j][i] = xopt[i] + signal[j][i]/sum(signal[j][i]*signal[j][i] for i in range(d))
            else:
                observed[j][i] = xopt[i]
            true[j][i] = xopt[i]
    return [signal, observed, true]


def draw_region(x, c, b, A, G, g):
    hull = ConvexHull(x[:,0:2])
    plt.plot(x[:, 0], x[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(x[simplex, 0], x[simplex, 1], 'r-')

    ###############################
    datanum = len(x)
    x_after = np.zeros((datanum, len(x[0])))
    for i in range(datanum):
        buffer = forward2A(c[i], b, A, G, g)
        for j in range(len(x[0])):
            x_after[i][j] = buffer[j]
    hull1 = ConvexHull(x_after[:,0:2])
    plt.plot(x_after[:, 0], x_after[:, 1], 'o')
    for simplex in hull1.simplices:
        plt.plot(x_after[simplex, 0], x_after[simplex, 1], 'b--')
    plt.axis('equal')
    plt.show()

if __name__=="__main__":
    dimension = 5
    e = np.ones(dimension)
    b = math.sqrt(2)

    (xopt, c) = synthetic_forward1(dimension, e, b)
