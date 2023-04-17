from scipy.linalg import expm
import numpy as np
import networkx as nx

"""
This code attemps to do with fixed time steps and somehow simple the code for learning heat
1st attempt will leave the H still
"""

def heat_random(N,M,S,se,p=0.2):
    """
    Create random
    X = NxM signal with se standard deviation
    L = NxN Laplacian
    H = NSxM sparse representation
    tau = 1xS diffusion coefficients
    """
    # create tau with entries between 0 and 4
    tau = []
    for s in range(S):
        tau.append(4*np.random.rand())
    # create graph, random Laplacian and dictionary
    rg = nx.fast_gnp_random_graph(N,p=p)
    L = nx.laplacian_matrix(rg).toarray()
    L = N*L/np.trace(L) # renormalize so trace(L)=N
    D = heat_dict(L,tau)
    # create sparse representation H
    random_atoms = []
    random_hs = []
    for m in range(M):
        random_atoms.append(np.random.choice(D.shape[1], 3, replace=False))
        random_hs.append(np.random.randn(3))
    xs = []
    H_ground = np.zeros((N*len(tau),M))
    for m, atom in enumerate(random_atoms):
        xs.append(np.squeeze(D[:,atom]@random_hs[m]))
        H_ground[atom,m] = random_hs[m]     
    # before adding noise
    X_clean = np.matrix(xs).T
    # noisy observations
    X = X_clean + np.sqrt(se)*np.random.randn(X_clean.shape[0],X_clean.shape[1])
    return X, L, H_ground, tau

def heat_dict(L,tau):
    # need to put axis=1 to do it horizontally
    return np.concatenate([expm(-tau_s*L) for tau_s in tau],axis=1)

def heat_grad(opt,X,H,L,tau):
    if opt==1:
        D = heat_dict(L,tau)
        return -2*D.T@(X-D@H)
    return

def heat_lip(opt,X,H,L,tau):
    if opt==1:
        D = heat_dict(L,tau)
        ct = np.linalg.norm(2*D.T@D,'fro')**2
        return ct
    return

def heat_learn(X,L,H,tau,alpha,beta,gamma1=1.1,gamma2=1.1,iter=100):
    # with fixed timestep and number of iterations
    # if L, H, tau are not initialized, pass it yourself
    for i in range(iter):
        # H update
        ct = gamma1*heat_lip(opt=1,X=X,H=H,L=L,tau=tau)
        H = H - heat_grad(opt=1,X=X,H=H,L=L,tau=tau)/ct
        H = np.multiply(np.sign(H),np.maximum(np.abs(H)-alpha/ct,0))
        # L update
        
    return L,H,tau