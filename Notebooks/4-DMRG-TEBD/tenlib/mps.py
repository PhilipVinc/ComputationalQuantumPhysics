import numpy as np
from numpy import linalg
from einops import rearrange
from typing import Optional
from opt_einsum import contract

from .utils import compact_svd, truncate, contract_left, contract_right

class MPS:
    N : int
    "Number of degrees of freedom"

    d : int
    "Local dimension"

    chi: int
    "Bond dimension"

    M: list[np.ndarray,...]
    """
    A list of N local tensors of size (chi,d,chi) in the bulk, otherwise (1, d, chi) or (chi, d, 1) at the edges
    """

    def __init__(self, N : int, d : int, chi : int, *, seed: Optional[int] = None, dtype: np.dtype = np.float64):
        """
        Initialises randomly the MPS matrices M.
        """
        self.N = N
        self.d = d
        self.chi = chi

        rng = np.random.default_rng(seed)

        self.M = [None for _ in range(N)]
        self.M[0]   = rng.normal(size=(1, d, chi))
        self.M[N-1] = rng.normal(size=(chi, d, 1))
        for i in range(1,N-1):
            self.M[i] = rng.normal(size=(chi, d, chi))

        # Singular-Values
        self.Svr = [0 for x in range(N+1)]
        self.Svr[0] = np.array([1])
        

    @classmethod
    def init_product_state(cls, N : int, d : int, chi : int = 1, state="up") -> "MPS":
        """
        Initialises an MPS in the trivial state `state`.
        """
        mps = cls(N, d, chi)

        for x in range(mps.N):
            mps.M[x].fill(0)
            if state == "up":
                mps.M[x][0,0,0] = 1
            elif state == "down":
                mps.M[x][0,1,0] = 1
        return mps
    
    @classmethod
    def from_statevector(cls, N : int, d : int, statevector : np.ndarray, *, 
                         rcond: float = None, chi : int | None = None) -> "MPS":
        """
        Initialise an MPS from a wavefunction.

        Args:
            chi: is a maximum cutoff
        """
        if statevector.size != d**N:
            raise ValueError(f"Statevector has size {statevector.size} but expected {d**N}")

        # convert to a single rank-N tensor with dimension(d,d,d,d,d,d)
        psi = statevector.reshape([d**(N-1), d])

        # list to store the MPS tensors
        #if chi is None:
        #
        #mps = cls(N, d, chi)
        M = [None for _ in range(N)]
        _chi_r = 1


        err = 0.0
        for i in range(N - 1, 0, -1):
            # from             to 
            # <     M     >   =   <     M     >
            #   | | | | |            ❚      |
            psi = psi.reshape((d**(i), d * _chi_r))
            
            # SVD
            U, S, V = compact_svd(psi)
            U, S, V, err_i = truncate(U, S, V, chi=None, rcond=rcond)
            err = err + err_i

            # 
            _chi_l = V.shape[0]
            assert d * _chi_r*_chi_l  == V.size
            M[i] = V.reshape((_chi_l, d, _chi_r))
            _chi_r = _chi_l
            
            US = U @ np.diag(S)
            psi = US

        M[0] = US.reshape((1, d, _chi_r))
        mps = cls(N, d, _chi_r)
        mps.M = M

        if err > 0:
            print("Truncation error is ", err)

        return mps
    
    @property
    def bond_dimensions(self) -> list[int]:
        bond_dimensions = tuple(self.M[i].shape[0] for i in range(self.N-1))
        return bond_dimensions + (self.M[-1].shape[2],)

    def normalize_right(self, chi: int | None = None, rcond: float | None = None, verbose: bool = True):
        if chi is not None and rcond is not None:
            raise ValueError("chi and rcond cannot be specified at the same time")
        elif chi is not None and chi <= 0:
            raise ValueError("chi must be positive")
        
        #   Index order
        #    0--M--2 
        #       |
        #       1

        "Ensure the state is in canonical form."
        err = 0
        for i in range(self.N - 1, 0, -1):
            Mi = self.M[i]
            _chi_r = Mi.shape[2]
            # Reshape using einops for clarity
            B_reshaped = rearrange(Mi, 'a b c -> a (b c)')
            U, S, V = compact_svd(B_reshaped)
            S = S / linalg.norm(S)

            # Truncate according to maximum chi or rcond
            U, S, V, err_i = truncate(U, S, V, chi, rcond)
            err += err_i
            self.Svr[i+1] = np.array(S)

            # Reshape V back and assign
            self.M[i] = V.reshape(S.size, Mi.shape[1], Mi.shape[2])
            
            # Update B[i-1] with einsum for clarity
            US = U @ np.diag(S)
            self.M[i - 1] = contract('abc,cd->abd', self.M[i - 1], US)

        # Final normalization for B[0]
        B0_reshaped = rearrange(self.M[0], 'a b c -> a (b c)')
        U, S, V = compact_svd(B0_reshaped)
        U, S, V, err_i = truncate(U, S, V, chi, rcond)
        err += err_i
        self.Svr[0+1] = np.array(S)

        self.M[0] = V.reshape(*self.M[0].shape)

        if verbose and err > 0:
            print(f"Truncated bond dimension with error: {err}")
    
    def normalize_left(self):
        """
        Normalize the MPS to the left.
        """
        for i in range(0, self.N - 1):
            M = self.M[i]
            _chi_l = M.shape[0]
            M_reshaped = rearrange(M, 'a b c -> (a b) c')
            U, S, V = compact_svd(M_reshaped)
            S = S / linalg.norm(S)
            self.M[i] = U.reshape(_chi_l, M.shape[1], S.size)
            SV = np.diag(S) @ V
            self.M[i + 1] = contract('ab,bcd->acd', SV, self.M[i + 1])

    def check_normalization(self, which='R'):
        if which == 'R':
            for i in range(self.N):
                X = contract('aib,cib->ac', self.M[i], self.M[i].conj())
                I_mat = np.eye(self.M[i].shape[0])
                if not np.allclose(X, I_mat):
                    return False
        elif which == 'L':
            for i in range(self.N):
                X = contract('aib,aic->bc', self.M[i].conj(), self.M[i])
                I_mat = np.eye(self.M[i].shape[2])
                if not np.allclose(X, I_mat):
                    return False
        # if we did not return false until now, return True!
        return True

    def compute_EntEntropy(self):
        Sent = np.zeros(self.N-1)
        Mlist = self.M.copy()
        
        for i in range(0, self.N-1):
            M = Mlist[i]
            shpM = M.shape
            
            U, S, V = linalg.svd(rearrange(M, 'a b c -> (a b) c'), full_matrices=False)
            S /= linalg.norm(S)
            Mlist[i] =  U.reshape(shpM[0], shpM[1], S.size)
            if i!= self.N-1:
                SV = np.diag(S)@V
                Mlist[i+1] = contract('ab,bcd->acd', SV, Mlist[i+1])

            S = S[S>1e-15]
            Sent[i] = np.sum(-S**2*np.log(S**2))
        return Sent

    def __repr__(self):
        return f"{type(self).__name__}(N={self.N}, d={self.d}, chi={self.chi}, normalized={self.check_normalization()})"
