import numpy as np
from numpy import linalg
from einops import rearrange
from typing import Optional
from opt_einsum import contract


def compact_svd(mat):
    #return np.linalg.svd(mat)
	U,S,V = linalg.svd(mat)
    
	if mat.shape[0] > mat.shape[1]:
		U = U[:,:mat.shape[1]]

	elif mat.shape[0] < mat.shape[1]:
		V = V[:mat.shape[0],:]

	return U,S,V
        
def truncate(U, S, V, chi, rcond):
    # Truncate according to maximum chi or rcond
    orig_bond_dim = S.size 
    err = 0
    if chi is not None and chi < S.size:
        err = err + np.sum(S[chi:])
        S = S[:chi]
    if rcond is not None:
        Snew = S[S >= rcond]
        if Snew.size != S.size:
            err = err + np.sum(S[Snew.size:])
        S = Snew
    
    if S.size != orig_bond_dim:
        U = U[:, :S.size]
        V = V[:S.size, :]
    return U, S, V, err

def contract_right(A,W,B,R):
    ## tensor contraction from the right hand side
    ##  0-+     -A--+
    ##    |      |  |
    ##  1-R' =  -W--R
    ##    |      |  |
    ##  2-+     -B--+    
    return contract("aiu, bvij, cjz, uvz -> abc", A, W, B, R)

def contract_left(A,W,B,L):
    ## tensor contraction from the left hand side
    ## +--   +--A-0         a --+-- c
    ## |     |  |               b
    ## L'- = L--W-1         d --+-- e
    ## |     |  |               f
    ## +--   +--B-2         g --+-- h
    return contract("abc, debf, gfh, adg -> ceh", A, W, B, L)
