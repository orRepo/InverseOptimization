
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import gurobipy as gp
from gurobipy import GRB
import numpy.random as rd
import math
import matplotlib.pyplot as plt
import numpy as np
import time

            

    
def generateData(fix, dim_f, datanum, noise, G2, g2):
 
    xiv=rd.uniform(0.2,1, (datanum, dim_f))
    opt_g=np.ones((datanum, dim_f))
    opt_obj=np.ones((datanum, 1))
    
    if fix:
        demand=np.ones((datanum, dim_f))
    else:
        demand = np.ones((datanum, dim_f))
        demand[:,0] = rd.uniform(0.3,1.5,(datanum,))
        demand[:,1] = rd.uniform(0.36,1.8,(datanum,))
        demand[:,2] = rd.uniform(0.42,2.1,(datanum,))
        demand[:,3] = rd.uniform(0.48,2.4,(datanum,))
        demand[:,4] = rd.uniform(0.54,2.7,(datanum,))
    
    for n in range(datanum):
        [opt_g[n,:],opt_obj[n,0]] = powersys(xiv[n,:],demand[n,:],noise, G2, g2)
        
    return (np.array(demand), np.array(xiv), np.array(opt_g), np.array(opt_obj))

def primitiveset(dim_f,dim_p):
    G = np.zeros((dim_p,dim_p//2))
    g = np.ones((dim_p,))*3.5
    for i in range(dim_p):
        for j in range(dim_p//2):
            if i==j:
                G[i][j]=1
            elif i>=10 and i%10 == j:
                G[i][j] = -1
    return (G,g)

#[c,optx, G, g] = forward()
def powersys(xiv, d, noise, G2, g2):
        
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n=5 #num of the arcs
    
    m=len(xiv)
    g= M1.addMVar(m, lb=0)
        
    f= M1.addMVar(n, lb= -3.5, ub =3.5)
        
    M1.setObjective(xiv @ g , GRB.MINIMIZE)
    
    M1.addConstr( G2 @ g <= g2 )
    
    M1.addConstr(g[0]+f[0]-f[1]>=d[0])
    M1.addConstr(f[1]+f[3]>=d[1])
    M1.addConstr(g[2]-f[0]-f[2]>=d[2])
    M1.addConstr(f[2]+f[4]-f[3]>=d[3])
    M1.addConstr(g[4]-f[4]>=d[4])

    M1.optimize()
    
    opt_g = g.X
        
    if noise:
        for j in range(m):
            opt_g[j] = opt_g[j]+rd.normal(0,0.1)
            #opt_g[j] = opt_g[j]+rd.laplace(0,0.1/math.sqrt(2))
    
    return [opt_g, M1.objVal]


def predictability(price, demand, solution, G,g,G2,g2):
    
    M = gp.Model("miqcp")
    M.setParam('TimeLimit', 3600)
    
    datanum = len(price)
    
    I= M.addMVar((5,10), vtype=GRB.BINARY)
    O= M.addMVar((5,10), vtype=GRB.BINARY)
    
    gamma=M.addMVar((datanum,5,), lb = -GRB.INFINITY)
    f = M.addMVar((datanum,10,), lb=-3.5, ub=3.5)
    
    
    lamb = M.addMVar((datanum,20,), lb=0)
    beta = M.addMVar((datanum,10,), lb=0)
    
    M.setObjective( sum(gamma[n,:] @ gamma[n,:]for n in range(datanum)), GRB.MINIMIZE)
    # every node can only have at most 4 arcs, we specify these arcs
    M.addConstr(sum(O[i,j] for i in range(5) for j in range(10))==sum(I[i,j] for i in range(5) for j in range(10)))
    M.addConstrs(O[0,j]==0 for j in range(4,10))
    M.addConstrs(I[0,j]==0 for j in range(4,10))
    M.addConstrs(O[1,j]==0 for j in [1,2,3,7,8,9])
    M.addConstrs(I[1,j]==0 for j in [1,2,3,7,8,9])
    M.addConstrs(O[2,j]==0 for j in [0,2,3,5,6,9])
    M.addConstrs(I[2,j]==0 for j in [0,2,3,5,6,9])
    M.addConstrs(O[3,j]==0 for j in [0,1,3,4,6,8])
    M.addConstrs(I[3,j]==0 for j in [0,1,3,4,6,8]) 
    M.addConstrs(O[4,j]==0 for j in [0,1,2,4,5,7])
    M.addConstrs(I[4,j]==0 for j in [0,1,2,4,5,7]) 
    M.addConstr(sum(O[i,:] for i in range(5))<=1)
    M.addConstr(sum(I[i,:] for i in range(5))==sum(O[i,:] for i in range(5)))
    #M.addConstr(I[3,9]==1)
    #M.addConstr(sum(O[i,j] for i in range(5) for j in range(10))<=5)# limit the number of arcs (for testing)
    
    
    M.addConstrs( solution[n,i] + gamma[n,i] == O[i,:]@f[n,:]-I[i,:]@f[n,:]+ demand[n,i] for n in range(datanum) for i in range(5))
    M.addConstrs( G2 @solution[n,:]+G2 @gamma[n,:] <= g2 for n in range(datanum))
    M.addConstrs(price[n,:]@solution[n,:]+price[n,:]@gamma[n,:]-price[n,:]@demand[n,:]
    +lamb[n,:]@g- sum(demand[n,i]*sum(beta[n,j]*G2[j,i] for j in range(10))for i in range(5))+beta[n,:]@g2<=0 for n in range(datanum))
    #beta[n,:] @ G2 @ demand[n,:]
    
    M.addConstrs(price[n,:]@O[:,i]-price[n,:]@I[:,i]+lamb[n,:]@G[:,i]+beta[n,:]@ G2 @O[:,i]==0
                 for n in range(datanum) for i in range(10))

    M.optimize()
    
    return [I.X, O.X, M.Runtime]


def suboptimality(price, demand, solution, G, g, G2, g2):
    M = gp.Model("miqcp")
    M.setParam('TimeLimit', 3600)

    datanum = len(price)

    I = M.addMVar((5, 10), vtype=GRB.BINARY)
    O = M.addMVar((5, 10), vtype=GRB.BINARY)

    gamma = M.addMVar((datanum, 5,), lb=-GRB.INFINITY)
    gammao = M.addMVar((datanum,))
    f = M.addMVar((datanum, 10,), lb=-3.5, ub=3.5)

    lamb = M.addMVar((datanum, 20,), lb=0)
    beta = M.addMVar((datanum, 10,), lb=0)

    M.setObjective(sum(gamma[n, :] @ gamma[n, :] for n in range(datanum))
                   + sum(gammao[n]**2 for n in range(datanum)), GRB.MINIMIZE)
    # every node can only have at most 4 arcs, we specify these arcs
    M.addConstr(sum(O[i, j] for i in range(5) for j in range(10)) == sum(I[i, j] for i in range(5) for j in range(10)))
    M.addConstrs(O[0, j] == 0 for j in range(4, 10))
    M.addConstrs(I[0, j] == 0 for j in range(4, 10))
    M.addConstrs(O[1, j] == 0 for j in [1, 2, 3, 7, 8, 9])
    M.addConstrs(I[1, j] == 0 for j in [1, 2, 3, 7, 8, 9])
    M.addConstrs(O[2, j] == 0 for j in [0, 2, 3, 5, 6, 9])
    M.addConstrs(I[2, j] == 0 for j in [0, 2, 3, 5, 6, 9])
    M.addConstrs(O[3, j] == 0 for j in [0, 1, 3, 4, 6, 8])
    M.addConstrs(I[3, j] == 0 for j in [0, 1, 3, 4, 6, 8])
    M.addConstrs(O[4, j] == 0 for j in [0, 1, 2, 4, 5, 7])
    M.addConstrs(I[4, j] == 0 for j in [0, 1, 2, 4, 5, 7])
    M.addConstr(sum(O[i, :] for i in range(5)) <= 1)
    M.addConstr(sum(I[i, :] for i in range(5)) == sum(O[i, :] for i in range(5)))
    # M.addConstr(I[3,9]==1)
    # M.addConstr(sum(O[i,j] for i in range(5) for j in range(10))<=5)# limit the number of arcs (for testing)

    M.addConstrs(
        solution[n, i] + gamma[n, i] == O[i, :] @ f[n, :] - I[i, :] @ f[n, :] + demand[n, i] for n in range(datanum) for
        i in range(5))
    M.addConstrs(G2 @ solution[n, :] + G2 @ gamma[n, :] <= g2 for n in range(datanum))
    M.addConstrs(price[n, :] @ solution[n, :] - price[n, :] @ demand[n, :] + lamb[n, :] @ g - sum(
        demand[n, i] * sum(beta[n, j] * G2[j, i] for j in range(10)) for i in range(5))
                 + beta[n, :] @ g2 <= gammao[n] for n in range(datanum))
    # beta[n,:] @ G2 @ demand[n,:]

    M.addConstrs(price[n, :] @ O[:, i] - price[n, :] @ I[:, i] + lamb[n, :] @ G[:, i] + beta[n, :] @ G2 @ O[:, i] == 0
                 for n in range(datanum) for i in range(10))

    M.optimize()

    return [I.X, O.X]

def enume_predictability(price, demand, solution, G, g, G2, g2, I, O):
    M = gp.Model("miqcp")
    M.setParam('TimeLimit', 3600)

    datanum = len(price)

    gamma = M.addMVar((datanum, 5,), lb=-GRB.INFINITY)
    gammao = M.addMVar((datanum,))
    f = M.addMVar((datanum, 10,), lb=-3.5, ub=3.5)

    lamb = M.addMVar((datanum, 20,), lb=0)
    beta = M.addMVar((datanum, 10,), lb=0)

    M.setObjective(sum(gamma[n, :] @ gamma[n, :] for n in range(datanum)), GRB.MINIMIZE)
    # every node can only have at most 4 arcs, we specify these arcs
    # M.addConstr(I[3,9]==1)
    # M.addConstr(sum(O[i,j] for i in range(5) for j in range(10))<=5)# limit the number of arcs (for testing)

    M.addConstrs(
        solution[n, i] + gamma[n, i] == O[i, :] @ f[n, :] - I[i, :] @ f[n, :] + demand[n, i] for n in range(datanum) for
        i in range(5))
    M.addConstrs(G2 @ solution[n, :] + G2 @ gamma[n, :] <= g2 for n in range(datanum))
    M.addConstrs(price[n, :] @ solution[n, :] + price[n, :] @ gamma[n, :] - price[n, :] @ demand[n, :]
                 + lamb[n, :] @ g - sum(
        demand[n, i] * sum(beta[n, j] * G2[j, i] for j in range(10)) for i in range(5)) + beta[n, :] @ g2 <= 0 for n in
                 range(datanum))
    # beta[n,:] @ G2 @ demand[n,:]

    M.addConstrs(price[n, :] @ O[:, i] - price[n, :] @ I[:, i] + lamb[n, :] @ G[:, i] + beta[n, :] @ G2 @ O[:, i] == 0
                 for n in range(datanum) for i in range(10))

    M.optimize()
    #print(M.Status)
    return [M.Status, M.Runtime]

def enumerate_condition(I,O):
    c1 = all([sum(O[i, j] for i in range(5) for j in range(10)) == sum(I[i, j] for i in range(5) for j in range(10))])
    c2 = all([O[0, j] == 0 for j in range(4, 10)])
    c3 = all([I[0, j] == 0 for j in range(4, 10)])
    c4 = all([O[1, j] == 0 for j in [1, 2, 3, 7, 8, 9]])
    c5 = all([I[1, j] == 0 for j in [1, 2, 3, 7, 8, 9]])
    c6 = all([O[2, j] == 0 for j in [0, 2, 3, 5, 6, 9]])
    c7 = all([I[2, j] == 0 for j in [0, 2, 3, 5, 6, 9]])
    c8 = all([O[3, j] == 0 for j in [0, 1, 3, 4, 6, 8]])
    c9 = all([I[3, j] == 0 for j in [0, 1, 3, 4, 6, 8]])
    c10 = all([O[4, j] == 0 for j in [0, 1, 2, 4, 5, 7]])
    c11 = all([I[4, j] == 0 for j in [0, 1, 2, 4, 5, 7]])
    c12 = all([all(sum(O[i, :] for i in range(5)) <= 1)])
    c13 = all([all(sum(I[i, :] for i in range(5)) == sum(O[i, :] for i in range(5)))])
    bool_arr = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13], dtype='bool')
    return bool_arr.all()


def Recovered(price, demand, O, I, G2, g2):
    M = gp.Model("miqcp")
    M.setParam('TimeLimit', 3600)
    f={}
    for i in range(10):
        f[i]= M.addVar(lb=-3.5, ub=3.5)
    m = len(price)
    g = M.addMVar(m, lb=0)

    M.setObjective(gp.quicksum(price[i]*g[i] for i in range(m)), GRB.MINIMIZE)

    M.addConstr(G2 @ g <= g2)
    # beta[n,:] @ G2 @ demand[n,:]

    M.addConstrs( gp.quicksum(O[i][j]*f[j] for j in range(10)) - gp.quicksum(I[i][j]*f[j] for j in range(10))
                  + demand[i] - g[i] <= 0 for i in range(m))
    M.optimize()
    return g.X


def newsys(xiv, d, noise, G2, g2):
        
    M1 = gp.Model("mip1")
    M1.setParam('TimeLimit', 3600)
    n=4 #num of the arcs
    
    m=len(xiv)
    g= M1.addMVar(m, lb=0)
        
    f= M1.addMVar(n, lb= -3.5, ub =3.5)
        
    M1.setObjective(xiv @ g , GRB.MINIMIZE)
    
    M1.addConstr( G2 @ g <= g2 )
    
    M1.addConstr(g[0]-f[0]>=d[0])
    M1.addConstr(f[1]>=d[1])
    M1.addConstr(g[2]-f[1]+f[3]>=d[2])
    M1.addConstr(f[2]>=d[3])
    M1.addConstr(g[4]-f[2]+f[0]-f[3]>=d[4])

    M1.optimize()
    
    opt_g = g.X
        
    if noise:
        for j in range(m):
            opt_g[j] = opt_g[j]+rd.normal(0,0.1)
            #opt_g[j] = opt_g[j]+rd.laplace(0,0.1/math.sqrt(2))
    
    return [opt_g, M1.objVal]
    


if __name__ == "__main__":
    R=5 # number of demand locations
    N=3 # number of power plants
    
    dim_p = int(R*(R-1)/2*2)    
    dim_f = R
    
    rd.seed(0)
    backtracking=1
    noise = 0
    
    fixdeman=0
    warmstart=1
    needdataset=1
    trainnum=50
    testnum=100
    
    G2 = np.zeros((2*R, R)) #capacity constraint for power plant
    g2 = np.zeros((2*R, ))
    for i in range(2*R):
        for j in range(R):
            if i==j:
                G2[i][j]=-1
            elif i>=R and i%5==j:
                G2[i][j]=1
        if i>=R and i%R in [0,2,4]:
            g2[i]=3.5
    
    if(needdataset):    
        (train_demand, train_price, train_opt_g, train_obj) = generateData(fixdeman, dim_f, trainnum, noise, G2, g2)
        (test_demand, test_price, test_opt, test_obj) = generateData(fixdeman, dim_f, testnum, noise, G2, g2)

    unique = np.unique(train_opt_g, axis=0)
    
    #define the primitive set #####################################################################
    
    (G,g) = primitiveset(dim_f,dim_p)


    ###################################################################################################

    if 0:
        [I,O, time1] = predictability(train_price, train_demand, train_opt_g, G, g, G2, g2)
        #[I, O] = suboptimality(train_price, train_demand, train_opt_g, G, g, G2, g2)
        test_recovered = {}
        for i in range(testnum):
            test_recovered[i] = Recovered(test_price[i], test_demand[i], O, I, G2, g2)
        error = sum((test_recovered[i][j] - test_opt[i][j]) ** 2 for i in range(testnum) for j in range(dim_f))
        print(f"error is {error}!")
        print(f'running time is {time1}.')
    else:
        now = time.time()
        for i in range(10):
            I_enu = np.random.randint(2, size=(5, 10)) # I
            O_enu = np.random.randint(2, size=(5, 10)) # O
            if 1: #enumerate_condition(I_enu, O_enu):
                [status, time2] = enume_predictability(train_price, train_demand, train_opt_g, G, g, G2, g2,I_enu, O_enu)
                print(f'Status is {status}')
        end = time.time()
        print(f'Running time of enumeration is {end-now}')




    # [residual, true_predict] = cL_p_true(test_price, test_demand, test_recovered, test_opt)
    # true_subopt = cL_sub_true(test_price, test_demand, test_recovered, test_opt)
    # out_predict = cL_pA(test_opt, test_price, G, g, 5, test_demand, C, A)
    # out_subopt = cL_sA(test_opt, test_price, G, g, 5, test_demand, C, A)
    # print(
    #     f"out_predict={out_predict}, true_predict = {true_predict}, out_subopt={out_subopt}, true_subopt = {true_subopt}.")

        


            
    
    
    

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
  
        
        










