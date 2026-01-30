import numpy as np
from scipy.linalg import orth 
import warnings
from typing import Callable

def lanczos_iteration(Afunc, vstart, numiter, ortho_info=False):
    """Perform a "matrix free" Lanczos iteration.

    Args:
        Afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - alpha:      diagonal real entries of Hessenberg matrix
          - beta:       off-diagonal real entries of Hessenberg matrix
          - V:          len(vstart) x numiter matrix containing the orthonormal Lanczos vectors
    """

    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart = vstart / nrmv

    alpha = np.zeros(numiter)
    beta  = np.zeros(numiter-1)

    V = np.zeros((numiter, len(vstart)), dtype=complex)
    V[0] = vstart

    for j in range(numiter-1):
        w = Afunc(V[j])
        alpha[j] = np.vdot(w, V[j]).real
        w -= alpha[j]*V[j] + (beta[j-1]*V[j-1] if j > 0 else 0)

        if ortho_info == ortho_info:
            for g in range(j+1):
                w -= np.vdot(w,V[g,:])*V[g,:]
        
        beta[j] = np.linalg.norm(w)
      
        if beta[j] < 100*len(vstart)*np.finfo(float).eps:
            warnings.warn(
                'beta[{}] ~= 0 encountered during Lanczos iteration.'.format(j),
                RuntimeWarning)
            # premature end of iteration
            numiter = j + 1
            return (alpha[:numiter], beta[:numiter-1], V[:numiter, :].T)

        V[j+1] = w / beta[j]
 
            
    j = numiter-1
    w = Afunc(V[j])
    alpha[j] = np.vdot(w, V[j]).real

    # complete final iteration
    return (alpha, beta, V.T)

def expm_krylov_lanczos(Afunc, v, dt, numiter, ortho_info = True): 
    """
    Compute Krylov subspace approximation of the matrix exponential
    applied to input vector: expm(dt*A)*v.

    Reference:
        M. Hochbruck and C. Lubich
        On Krylov subspace approximations to the matrix exponential operator
        SIAM J. Numer. Anal. 34, 1911 (1997)
    """

    alpha, beta, V = lanczos_iteration(Afunc, v, numiter, ortho_info=ortho_info)

    # diagonalize Hessenberg matrix
    w_hess, u_hess = np.linalg.eigh(np.diag(alpha) + np.diag(beta,1) + np.diag(beta,-1))

    return np.dot(V, np.dot(u_hess, np.linalg.norm(v) * np.exp(dt*w_hess) * u_hess[0]))

def optimize_lanczos(Afunc: Callable,v:np.ndarray,numiter: int):
    """
    Solve an eigenvalue problem using the Lanczos algorithm.

    .. example::

        ## tensor contraction for minimization
        ##         +--.--+
        ##         |  |  |
        ## H_eff = L--H--R
        ##         |  |  |
        ##         +--M--+       
        def apply_Heff(L,H,R,M):
            return contract(...) # tensor contraction according to the above scheme

        def local_minimization(M,L,H,R,Lsteps=10):
            Afunc = lambda x: apply_Heff(L, H, R, x.reshape(M.shape)).ravel()
            return optimize_lanczos(Afunc, M.ravel(), Lsteps)

    Args:
        Afunc (Callable): a function that takes a vector and returns the matrix-vector product of the matrix of interest with that vector
        v (np.ndarray): an initial vector
        numiter (int): the number of iterations to perform

    """

    alpha, beta, V = lanczos_iteration(Afunc, v, numiter, ortho_info=True)
    eig, w = np.linalg.eigh(np.diag(alpha) + np.diag(beta,1) + np.diag(beta,-1))
    return V@w[:,0], eig[0]