import numpy as np
from numpy import linalg
from einops import rearrange
from typing import Optional
from opt_einsum import contract

from .utils import contract_left, contract_right
from .mps import MPS

class MPO:
    def __init__(self,N,d):
        #   Index order
        #       2
        #       |
        #    0--W--1
        #       |
        #       3
        #
        # L: length of the tensor train
        # d: local Hilbert Space dimension
        self.N = N
        self.d = d
        self.W = [0 for x in range(N)]
    
    def __repr__(self):
        return f"{type(self).__name__}(L={self.N}, d={self.d})"

    def expect(self, wf: MPS):
        # M-M-M-M-M
        # | | | | |
        # W-W-W-W-W
        # | | | | |
        # M-M-M-M-M    
        if(wf.N != self.N): 
           raise Exception('MPS MPO length are different')
        Rtemp = np.ones((1,1,1),dtype=np.float64)
        for i in range(self.N-1,0,-1):
            Rtemp = contract_right(wf.M[i], self.W[i], wf.M[i].conj(), Rtemp)
        Rtemp = contract_right(wf.M[0], self.W[0], wf.M[0].conj(), Rtemp)
        return Rtemp[0][0][0]
