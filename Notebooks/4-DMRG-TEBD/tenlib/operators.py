import numpy as np

from .mpo import MPO

def Mz_global(L):
    """
    Constructs an MPO for the total magnetization along z
    """
    MzMPO = MPO(L,2)
    mz  = np.zeros((2,2,2,2)); mzl = np.zeros((1,2,2,2)); mzr = np.zeros((2,1,2,2))
    sigma_z  = np.array([[1, 0], [0,-1]]);  Id  = np.array([[1, 0], [0, 1]])
    # Bulk
    mz[0,0,:,:] = Id; mz[1,0,:,:] = sigma_z
    mz[1,1,:,:] = Id
    # Boundary
    mzr[0,0,:,:] = Id; mzr[1,0,:,:] = sigma_z
    mzl[0,0,:,:] = sigma_z; mzl[0,1,:,:] = Id    
    # Set the MPO
    MzMPO.W[0] = mzl; MzMPO.W[L-1] = mzr
    for i in range(1,L-1):
        MzMPO.W[i] = mz
    return MzMPO

def Mx_global(L: int) -> MPO:
    """
    Constructs an MPO for the total magnetization along x
    """
    MxMPO = MPO(L,2)
    mx  = np.zeros((2,2,2,2)); mxl = np.zeros((1,2,2,2)); mxr = np.zeros((2,1,2,2))
    sigma_x  = np.array([[0, 1], [1, 0]]);  Id  = np.array([[1, 0], [0, 1]])
    # Bulk
    mx[0,0,:,:] = Id; mx[1,0,:,:] = sigma_x
    mx[1,1,:,:] = Id
    # Boundary
    mxr[0,0,:,:] = Id; mxr[1,0,:,:] = sigma_x
    mxl[0,0,:,:] = sigma_x; mxl[0,1,:,:] = Id    
    # Set the MPO
    MxMPO.W[0] = mxl; MxMPO.W[L-1] = mxr
    for i in range(1,L-1):
        MxMPO.W[i] = mx
    return MxMPO

def Mz_local(L: int, i: int) -> MPO:
    """
    Constructs an MPO for the local magnetization along z at site i.
    """
    LMz = MPO(L, 2)
    sigma_z  = np.array([[1, 0], [0,-1]]);  Id  = np.array([[1, 0], [0, 1]])
    mzl = np.zeros((1,2,2,2)); mzr = np.zeros((2,1,2,2))
    mzr[0,0,:,:] = Id; 
    mzr[1,0,:,:] = Id;
    mzl[0,0,:,:] = Id; 
    mzl[0,1,:,:] = Id    
    
    LMz.W[0] = mzl; 
    LMz.W[L-1] = mzr
    for j in range(1,L-1):
        mz  = np.zeros((2,2,2,2));
        mz[0,0,:,:] = Id; 
        if j == i:
            mz[1,0,:,:] = sigma_z
        else:
            mz[1,0,:,:] = Id
        mz[1,1,:,:] = Id
        LMz.W[j] = mz
    return LMz


def Ising(L:int, h: float, J:float=1):
    """
    Constructs the Ising MPO Object
    """
    Ham = MPO(L,2)
        
    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the local bulk MPO
    H = np.zeros([3,3,2,2])
    H[0,0,:,:] = Id; H[2,2,:,:] = Id; H[2,0,:,:] = -h*Sx
    H[1,0,:,:] = Sz; H[2,1,:,:]= J*Sz

    # Building the boundary MPOs
    HL = np.zeros((1,3,2,2))
    HL[0,:,:,:] = H[2,:,:,:]
    HR = np.zeros((3,1,2,2))
    HR[:,0,:,:] = H[:,0,:,:]

    Ham.W[0] = HL
    Ham.W[L-1] = HR
    for i in range(1,L-1):
        Ham.W[i] = H
    return Ham

