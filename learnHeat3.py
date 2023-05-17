import numpy as np
import cvxpy as cp
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from scipy.linalg import expm
from scipy.spatial import distance_matrix
from scipy.sparse import random
from tqdm.notebook import tqdm
from functools import reduce
import networkx as nx
import pandas as pd
import pygsp as pg # this is better than networkx
import scores_table as scort # this is for main_table.ipynb
import itertools # this is for combinations of alpha and beta

# Lipshitz constant C1
def lipschitz_c1(L, tau):
    ''' 
    lipschitz_c1(L, tau)
    This function produce the lipschitz constant c1 from Laplacian matrix L and tau.
    '''
    return np.linalg.norm(2*D(L,tau).T@D(L,tau), 'fro')

def descent_condition_by_cost(X, Ltp1, Lt, Htp1, taut, alpha, beta):
    left = cal_cost(X, Ltp1, Htp1, taut, alpha, beta)
    right = cal_cost(X, Lt, Htp1, taut, alpha, beta)
    
    condition = left <= right
    return condition, {"cost_Ltp1":left,"cost_Lt":right}

def descent_condition(X, Ltp1, Lt, Htp1, taut, c2):
    left = Z(X, Ltp1, Htp1, taut)
    right = Z(X, Lt, Htp1, taut) + \
                matrix_inner(Ltp1-Lt,gradient_z_to_L(Lt, X, Htp1, taut)) + \
                c2/2*(np.linalg.norm(Ltp1-Lt, 'fro')**2)
    
    
    condition = left <= right
    return condition, (left,right)

def back_tracking(X, L_ground, Lt, Htp1, taut, gamma2, alpha, beta, verbose):
    N = X.shape[0]
    S = len(taut)
    eta = 1.1
    c2 = 0.01
    k = 1
    cond = False
    gradient = gradient_z_to_L(Lt, X, Htp1, taut)
    while cond == False:
        c2 = (eta**k)*c2
        dt = gamma2*c2
        Ltp1 = admm(X, Lt, gradient, Htp1, taut, dt, beta, verbose)
        k += 1
        cond, detail = descent_condition(X = X, Ltp1 = Ltp1, Lt = Lt, Htp1 = Htp1, 
                                         taut = taut, c2 = c2)
    for i in range(N):
        for j in range(N):
            if L_ground[i,j] == 0:
                Ltp1[i,j] = 0
    return Ltp1

# Lipshitz 
def lipschitz_c3(L, X, H, tau):
# TODO: Consider not to use hessian method..
#     N = L.shape[0]
#     S = len(tau)
#     H_list = H_matrix_to_list(H, N, S)
    
#     cost = []
#     for s, Hs in enumerate(H_list):
#         cost.append(2*np.linalg.norm(Hs, 'fro')*np.linalg.norm(X, 'fro')\
#                     +4*np.linalg.norm(Hs, 'fro')*sum([np.linalg.norm(Hsp, 'fro') for Hsp in H_list]))
#     return np.max(cost)*np.linalg.norm(L,2)**2
    Hessian = hessian_Z_to_tau(X, L, H, tau)
    return np.linalg.norm(Hessian,2)

def soft_threshold(Ht, Lt, X, taut, ct, alpha):
    G = Ht - 1/ct*gradient_z_to_H(Lt, X, Ht, taut)
    Htp1 = np.multiply(np.sign(G), np.maximum(np.abs(G)-alpha/ct, 0))
    return Htp1

def laplacian_to_adjacency(L):
    W = -np.copy(L)
    np.fill_diagonal(W,0)
    return W

def laplacian_to_vec(L):
    W = laplacian_to_adjacency(L)
    vec = np.tril(W,k=-1).flatten()
    return vec

def gradient_z_to_H(L, X, H, tau):
    return -2*D(L, tau).T@(X-D(L, tau)@H)

def gradient_z_to_L(L, X, H, tau):
    N = L.shape[0]
    S = len(tau)
    H_list = H_matrix_to_list(H, N, S)
    on_signals = [(-2)*dtrAenLdL(L, A = Hs@X.T, nu = -tau[s]) for s, Hs in enumerate(H_list)]
    off_signal = np.zeros(L.shape)
    for s, Hs in enumerate(H_list):
        off_signals = [dtrAenLdL(L, A = Hsp@Hs.T, nu= -(tau[s]+tau[sp])) for sp, Hsp in enumerate(H_list)]
        off_signal = off_signal + reduce(np.add, off_signals)
    on_signal = reduce(np.add, on_signals)
    g = on_signal + off_signal
    return g

def gradient_z_to_tau(L, X, H, tau):
    N = L.shape[0]
    S = len(tau)
    H_list = H_matrix_to_list(H, N, S)
    g = np.zeros(len(tau))
    for s, Hs in enumerate(H_list):
        on_signal = 2*np.trace(Hs@X.T@L@expm(-tau[s]*L))
        off_signals = [np.trace(Hsp@Hs.T@L@expm(-(tau[s]+tau[sp])*L)) for sp, Hsp in enumerate(H_list)]
        g[s] = on_signal -2*np.sum(off_signals)
    return g

def H_matrix_to_list(H, N, S):
    return np.split(H, [i*N for i in range(1,S)], axis=0)

def dtrAenLdL(L, A, nu):
    if nu == 0:
        return np.zeros(L.shape)
    return nu*dtrAeLdL(L = nu*L, A = A)

def dtrAeLdL(L, A):
    # eigen decomposition
    Eval, Evec = np.linalg.eig(L)
    Eval = np.real(Eval)
    Evec = np.real(Evec)
    # Define B
    B = np.zeros(L.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if Eval[i] == Eval[j]:
                B[i,i] = np.exp(Eval[i])
            else:
                B[i,j] = (np.exp(Eval[i])-np.exp(Eval[j])) / (Eval[i]-Eval[j])
    
    # derivative with respect to L
    return Evec@(np.multiply(Evec.T@A.T@Evec,B))@Evec.T

def cover_zeros(L_ground, percentage):
    """
    This function will cover a percentage of zeros in L_ground by 1 so it is
    not included as a priori knowledge
    """
    L_covered = np.copy(L_ground)
    num_zeros = np.count_nonzero(L_covered == 0)
    num_to_cover = int(np.ceil(percentage * num_zeros))
    zero_indices = np.argwhere(L_covered == 0)
    np.random.shuffle(zero_indices)
    for i in range(num_to_cover):
        idx = tuple(zero_indices[i])
        L_covered[idx] = 1
    return L_covered

def compare_zeros(L_learned, L_ground):
    """
    This function sees if learned negatives is true negatives
    """
    mask = (L_ground == 0)
    all_negatives = np.count_nonzero(mask)
    true_negatives = np.logical_and(L_learned == 0, mask)
    num_true_negatives = np.count_nonzero(true_negatives)
    return num_true_negatives/all_negatives

def admm(X, Lt, gradient, Htp1, taut, dt, beta, verbose):
    S = len(taut)
    # variable
    L = cp.Variable(Lt.shape)
    N = Lt.shape[0]
    # constraints
    constraints = [cp.trace(L)== N, L.T==L, L@np.ones((N,1))==np.zeros((N,1))]
    for i in range(N-1):
        constraints += [L[i][i+1:]<=0]
    # objective

    obj = cp.Minimize(cp.trace(gradient.T@(L-Lt)) \
                      + dt/2*(cp.norm(L-Lt, 'fro')**2) + beta*(cp.norm(L, 'fro')**2))
    # solve problem
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=verbose, solver=cp.SCS, scale = 1000, use_indirect = False)
    if L.value is None:
        prob.solve(verbose=verbose, solver=cp.MOSEK)
    return L.value

def Z(X, L, H, tau):
    return np.linalg.norm(X-D(L, tau)@H, 'fro')**2

def tautp1_closed(taut, X, Ltp1, Htp1, et):
    return np.maximum(np.array(taut)-gradient_z_to_tau(Ltp1, X, Htp1, taut)/et, 0)

def D(L, tau):
    return np.concatenate([expm(-tau_s*L) for tau_s in tau], axis=1)

def matrix_inner(A, B):
    return np.trace(B.T@A)

def hessian_Z_to_tau(X, L, H, tau):
    S = len(tau)
    N = L.shape[0]
    
    # Initialize
    Hessian = np.zeros((S,S))
    H_list = H_matrix_to_list(H, N, S)
    for s in range(S):
        Hs = H_list[s]
        for sp in range(S):
            Hsp = H_list[sp]
            if s == sp:
                Hessian[s,s] = -2*np.trace(Hs@X.T@L@L@expm(-tau[s]*L)) + 4*np.trace(Hs@Hs.T@L@L@expm(-tau[s]*L))\
                                + 2* (np.sum([np.trace(H_list[ss]@Hs.T@L@L@expm(-(tau[s]+tau[ss])*L)) for ss in range(S)])\
                                      -np.trace(Hs@Hs.T@L@L@expm(-(tau[s]+tau[s])*L)))
            else:
                Hessian[s,sp] = 2*np.trace(Hsp@Hs.T@L@L@expm(-(tau[s]+tau[sp])*L))
    return Hessian
    
def cal_cost(X, L, H, tau, alpha, beta):
    return Z(X, L, H, tau) + alpha*np.sum(np.abs(H)) + beta*(np.linalg.norm(L,'fro')**2)

def learn_heat(X,
               L_ground, 
               L0 = np.array([]), 
               H0 = np.array([]), 
               tau0 = [1,2,3], 
               alpha = 0.1, 
               beta = 0.1, 
               gamma1 = 1.1, 
               gamma2 = 1.1, 
               gamma3 = 1.1, 
               max_iter = 100, 
               verbose=False):
    '''
    learn_heat(X, L0, H0, tau0, alpha, beta, gamma1, gamma2, gamma3, max_iter, verbose)
    
    X: Data matrix. Each column should be an observation
    
    <Optional variables>
    L0: Initial matrix of laplacian matrix
    H0: Initial sparse coefficients
    tau0: Initial taus
    alpha : Sparsity regularization parameter
    beta: Sparsity regularization parameter
    gamma1, gamma2, gamma3: lipshitz constaint factor. Should be larger than 1
    max_iter: Maximum number of iteration. Default is 100.
    verbose: {True, Fale}. In order to see detailed log for all optimization step, please set as True.
    '''
    
    tautp1 = tau0
    N = X.shape[0]
    S = len(tautp1)
    # Initialize laplacian matrix from adjacency matrix
    if L0.shape[0] == 0:
        W = np.random.rand(N,N)
        W = W.T@W
        np.fill_diagonal(W, 0)
        Ltp1 = csgraph.laplacian(W, normed=False)
        Ltp1 = Ltp1/np.trace(Ltp1)*N
    else:
        Ltp1 = L0
        
    if H0.shape[0] == 0:
        Htp1 = np.random.rand(D(Ltp1, tautp1).shape[1], X.shape[1])
    else:
        Htp1 = H0
        
    # pbar = tqdm(range(0, max_iter), desc="Learning progress")
    pbar = tqdm(range(0, max_iter), desc="Learning progress",disable=True)
    #cost_bar = tqdm(total=0, position=1, bar_format='{desc}')
    cost_bar = tqdm(total=0, position=1, bar_format='{desc}',disable=True)
    steps = []
    costs = []
    for t in pbar:
        Lt = Ltp1
        taut = tautp1
        Ht = Htp1    
        # Step for choose c1 (lipschitz)
        ct = gamma1*lipschitz_c1(Lt, taut)
        
        # Update H
        Htp1 = soft_threshold(Ht, Lt, X, taut, ct, alpha)
        cost_bar.set_description_str(f"COST: {'{:0.5f}'.format(cal_cost(X, Lt, Htp1, taut, alpha, beta))} (H updated)")
        # Step to update L and D
        # Update L
        Ltp1 = back_tracking(X, L_ground, Lt, Htp1, taut, gamma2, alpha, beta, verbose)
        cost_bar.set_description_str(f"COST: {'{:0.5f}'.format(cal_cost(X, Ltp1, Htp1, taut, alpha, beta))} (L updated)")
        
        ## Step to update tau and D
        et = gamma3*lipschitz_c3(Ltp1, X, Htp1, taut)
        tautp1 = tautp1_closed(taut, X, Ltp1, Htp1, et)
        cost_bar.set_description_str(f"COST: {'{:0.5f}'.format(cal_cost(X, Ltp1, Htp1, tautp1, alpha, beta))} (tau updated)")
        steps.append(t)
        costs.append(cal_cost(X, Ltp1, Htp1, tautp1, alpha, beta))
        
    W_learn = -Ltp1
    np.fill_diagonal(W_learn, 0)
    
    learning_curve = {"step": steps, "cost": costs}
    result={"L": Ltp1, "H": Htp1, "tau": tautp1, "W": W_learn, "learning_curve":learning_curve}
    
    return result

def create_signal(N=20,p=0.2,tau_ground=[2.5,4],M=100,se=0):
    rg = nx.fast_gnp_random_graph(N,p)
    L_ground = nx.laplacian_matrix(rg).toarray()
    D_ground = D(L_ground, tau_ground)
    random_atoms = []
    random_hs = []
    for m in range(M):
        random_atoms.append(np.random.choice(D_ground.shape[1], 3, replace=False))
        random_hs.append(np.random.randn(3))
    xs = []
    H_ground = np.zeros((N*len(tau_ground),M))
    for m, atom in enumerate(random_atoms):
        xs.append(np.squeeze(D_ground[:,atom]@random_hs[m]))
        H_ground[atom,m] = random_hs[m]
    X_clean = np.matrix(xs).T
    X = X_clean + np.sqrt(se)*np.random.randn(X_clean.shape[0],X_clean.shape[1])
    return X, L_ground, H_ground, tau_ground

def create_signal2(N=20,tau_ground=[2.5,4],M=100,se=0,kappa=0.75,sigma=0.5):
    # this time creates RBF graph
    coordinates = np.random.rand(N,2)
    dist_mat = distance_matrix(coordinates,coordinates)
    dist_mat[dist_mat<kappa] = 0
    L_ground = -np.where(dist_mat!=0,np.exp(-dist_mat**2/(2*sigma**2)),0)
    np.fill_diagonal(L_ground,-np.sum(L_ground,axis=1))
    ###########################################################
    D_ground = D(L_ground, tau_ground)
    random_atoms = []
    random_hs = []
    for m in range(M):
        random_atoms.append(np.random.choice(D_ground.shape[1], 3, replace=False))
        random_hs.append(np.random.randn(3))
    xs = []
    H_ground = np.zeros((N*len(tau_ground),M))
    for m, atom in enumerate(random_atoms):
        xs.append(np.squeeze(D_ground[:,atom]@random_hs[m]))
        H_ground[atom,m] = random_hs[m]
    X_clean = np.matrix(xs).T
    X = X_clean + np.sqrt(se)*np.random.randn(X_clean.shape[0],X_clean.shape[1])
    return X, L_ground, H_ground, tau_ground

def create_deltas(L,taus,se=0):
    """
    Inputs:
    L: Laplacian
    tau_ground: diffusion processes
    Output:
    X: signal of evolved deltas
    """
    N = L.shape[0]
    H = np.eye(N)
    signals = []
    X = np.concatenate([expm(-tau*L)@H for tau in taus],axis=1)
    X += np.random.randn(X.shape[0],X.shape[1])*se
    return X

def create_deltas2(L,taus,se=0):
    """
    Inputs:
    L: Laplacian
    tau_ground: diffusion processes
    Output:
    X: signal of evolved deltas
    """
    N = L.shape[0]
    H = np.eye(N)
    Hfull = random(N,N).todense()
    X = np.concatenate([expm(-tau*L)@H for tau in taus],axis=1)
    X2 = np.concatenate([expm(-tau*L)@Hfull for tau in taus],axis=1)
    X3 = np.concatenate([X,X2],axis=1)
    X3 += np.random.randn(X3.shape[0],X3.shape[1])*se
    return X3

def heat_scores(L1,L2,num_trials=10):
    # L1 is the learned Laplacian, L2 the ground one
    precisions = []
    recalls = []
    # vectorize matrices
    # there needs to be some kind of normalization
    W1 = np.tril(-L1,k=-1).flatten()
    W1[W1<0]=0
    W2 = np.tril(-L2,k=-1).flatten()
    W2 = (W2>0).astype(int)
    # normalization
    W1 = W1/np.max(W1)
    thresholds = np.linspace(0,1,num_trials)
    for threshold in thresholds:
        W_temp = (W1>=threshold).astype(int)
        tp = np.sum((W_temp==1)&(W2==1))
        fp = np.sum((W_temp==1)&(W2==0))
        fn = np.sum((W_temp==0)&(W2==1))
        if tp+fp==0:
            precision = 1
        else:
            precision = tp/(tp+fp)
        if tp+fn==0:
            recall = 1
        else:
            recall = tp/(tp+fn)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls

def heat_list(type, N, se, alpha, beta, max_iter, tau_ground = [1,2.5,4]):
    """
    Will generate a random graph of type=type of N vertices, create a signal X
    given by deltas over the graph of size M=NS and max_iter=40
    """
    # create the graph
    if type==1:
        _, L, _, _ = create_signal2(N,tau_ground=tau_ground)
        W = laplacian_to_adjacency(L)
        G = pg.graphs.Graph(W)
    elif type==2:
        G = pg.graphs.ErdosRenyi(N,0.4)
    elif type==3:
        G = pg.graphs.BarabasiAlbert(N)
   
    L_ground = G.L.todense()
    # create the signal
    X = create_deltas2(L_ground, taus=tau_ground, se=se)

    # learn the graph from the signal
    res = learn_heat(X,tau0=[1,2,3],alpha=alpha,beta=beta,max_iter=max_iter)

    # extract vectorized laplacians
    ground = -np.tril(L_ground,k=-1).flatten()
    learned = -np.tril(res["L"],k=-1).flatten()

    # create a list with the scores
    _, best_precision_recall = scort.threshold_precision_recall(learned,ground)
    best_precision_recall.append(scort.mean_squared_error(learned,ground))
    return best_precision_recall

def heat_sim(sim=4, type=1, N=20, se=0, alphav=[0.1,0.05,0.01], betav=[0.05,0.01], max_iter=30, tau_ground = [1,2.5,4]):
    """
    Run sim simulations for heat_list and take the average and the standard deviation
    """
    observations = []
    for comb in itertools.product(alphav,betav):
        for i in range(sim):
            observations.append(heat_list(type, N, se, comb[0], comb[1], max_iter, tau_ground = [1,2.5,4]))
    X = np.array(observations)
    heat_mean = np.mean(X,axis=0)
    heat_stdev = np.std(X,axis=0)
    # i need to get the row with the maximum value in column 2 (fmeasure)
    max_index = np.argmax(X[:, 2])
    heat_max = X[max_index,:]
    return heat_mean, heat_stdev, heat_max

def heat_master(N,max_iter):
    """
    Here we cycle in type and se and we need to 
    """
    print("entered function")
    type = [1,2,3]
    se = [0,0.1,0.5]
    data = ['Graph Type','\sigma noise','Information','Precision','Recall','F score','NMI','l2error']
    df = pd.DataFrame(columns=data)
    for comb in itertools.product(type,se):
        # here we do our stuff
        print("type is {}, standard deviation is {}, here we go".format(comb[0],comb[1]))
        mean, stdev, maxx = heat_sim(type=comb[0],N=N,se=comb[1],max_iter=max_iter)
        data1 = {'Graph Type':comb[0],'\sigma noise':comb[1],'Information':'Mean','Precision':mean[0],'Recall':mean[1],'F score':mean[2],'NMI':mean[3],'l2error':mean[4]}
        data2 = {'Graph Type':comb[0],'\sigma noise':comb[1],'Information':'Stdev','Precision':stdev[0],'Recall':stdev[1],'F score':stdev[2],'NMI':stdev[3],'l2error':stdev[4]}
        data3 = {'Graph Type':comb[0],'\sigma noise':comb[1],'Information':'Max','Precision':maxx[0],'Recall':maxx[1],'F score':maxx[2],'NMI':maxx[3],'l2error':maxx[4]}
        df.loc[len(df)] = data1
        df.loc[len(df)] = data2
        df.loc[len(df)] = data3

    return df

def filter_by_edges(L,N):
    # Assume L is your Laplacian matrix
    # Convert L to a weighted adjacency matrix W
    n_nodes = L.shape[0]
    D = np.diag(np.sum(L, axis=1))
    W = D - L

    # Sort the edges of the graph by weight
    sorted_edges = np.argsort(W.ravel())[::-1]

    # Set the weights of the first N edges to their original values, and set the rest to zero
    keep_edges = sorted_edges[:N]
    W_new = np.zeros_like(W)
    W_new.ravel()[keep_edges] = W.ravel()[keep_edges]

    # Construct a new Laplacian matrix L_new using the modified weighted adjacency matrix W
    return W_new

##################################################################################

def heat_persistent(L,n_trials=1000,reverse=False):
    """
    This clumsy function tries to return the most persistant topology over the thresholds
    """
    Lnew = np.copy(L)
    times = np.linspace(0,1,n_trials)
    changed = np.zeros(n_trials)
    current = heat_numedges(Lnew)
    counter = 0
    for t in range(n_trials):
        Lnew = heat_threshold(L,times[t])
        new_edges = heat_numedges(Lnew)
        counter +=1
        if new_edges != current:
            changed[t] += counter
            current = new_edges
            counter = 0


    # get boolean array indicating non-zero elements
    non_zero = changed != 0

    # get indices of non-zero elements that are not the first or last
    valid_idx = np.where(non_zero & (np.arange(len(changed)) != 0) & (np.arange(len(changed)) != len(changed) - 1))[0]

    # get index of largest non-zero element that satisfies conditions
    if reverse==False:
        closest_argmin = valid_idx[np.argmax(changed[valid_idx])]
    else:
        try:
            min_val = min(changed[valid_idx])
            argmin_vals = [i for i, val in enumerate(changed[valid_idx]) if val == min_val]
            middle_idx = len(changed[valid_idx]) // 2
            if len(argmin_vals) == 1:
                closest_argmin = argmin_vals[0]
            else:
                closest_argmin = min(argmin_vals, key=lambda x: abs(x-middle_idx))
        except ValueError:
            closest_argmin = 0
    optimal_threshold = heat_threshold(L,times[closest_argmin-1])
    # compute optimal threshold
    return optimal_threshold, changed

def heat_threshold(L,thresh,load=False):
    """
    Given Laplacian, set to 0 all off-diagonal entries below a certain threshold
    If load is set to true then load all the lost weights to the existing weights
    """
    Lcopy = np.copy(L)
    D = np.diag(np.diag(Lcopy))
    np.fill_diagonal(Lcopy,0)
    total_weight = np.sum(np.sum(Lcopy[Lcopy > -thresh]))
    Lcopy[Lcopy > -thresh] = 0
    # need number of non-thresholed entries
    if load==True:
        L2copy = np.copy(Lcopy)
        L2copy[L2copy < 0] = -1
        outside_thresh = np.sum(np.sum(L2copy[L2copy<0]))
        Lcopy[Lcopy < 0] -= total_weight/outside_thresh

    return Lcopy+D

def heat_numedges(L):
    Lcopy = np.copy(L)
    Lcopy[Lcopy < 0] = -1
    return -np.sum(np.sum(Lcopy[Lcopy<0]))/2

##########################################################
# THE NORMALIZED GRAPHS

def normalized_L(L):
    return (L.shape[0])*L/np.trace(L)

def heat_graph_ER(N,p=0.3):
    graph = nx.erdos_renyi_graph(N,p)
    L = nx.laplacian_matrix(graph).toarray()
    L = L/np.trace(L)*N
    return L

def heat_graph_RBF(N,kappa=0.75,sigma=0.5):
    _, L, _, _ = create_signal2(N,kappa=kappa,sigma=sigma)
    return normalized_L(L)