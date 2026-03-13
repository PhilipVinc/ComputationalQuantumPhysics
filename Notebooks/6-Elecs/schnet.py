from flax import linen as nn
import jax
import jax.numpy as jnp


# Implementing this network by yourself might take a bit of time, so
def cosine_cutoff(r, r_cut):
    '''Smooth cutoff: 1 at r=0, 0 at r>=r_cut, C1 everywhere.'''
    return jnp.where(r < r_cut, 0.5 * (jnp.cos(jnp.pi * r / r_cut) + 1.0), 0.0)


class RBFLayer(nn.Module):
    '''Radial Basis Function expansion: distances -> feature vectors.'''
    n_rbf: int   = 20
    r_cut: float = 6.0

    @nn.compact
    def __call__(self, r):
        # r: (n_pairs,) — interatomic distances in Bohr
        centers = jnp.linspace(0.5, self.r_cut, self.n_rbf)
        gamma   = (self.n_rbf / (self.r_cut - 0.5)) ** 2
        rbf     = jnp.exp(-gamma * (r[:, None] - centers[None, :]) ** 2)
        cutoff  = cosine_cutoff(r, self.r_cut)[:, None]
        return rbf * cutoff   # (n_pairs, n_rbf); exactly 0 beyond r_cut


class InteractionBlock(nn.Module):
    '''One SchNet message-passing layer.'''
    n_features: int

    @nn.compact
    def __call__(self, h, rbf_ij, idx_i, idx_j):
        n_atoms = h.shape[0]

        # Filter network: RBF features -> weights for each neighbour message
        W = nn.Dense(self.n_features)(rbf_ij)
        W = nn.silu(W)
        W = nn.Dense(self.n_features)(W)         # (n_pairs, F)

        # Messages: h_j weighted by distance-dependent filter
        msgs = h[idx_j] * W                      # (n_pairs, F)

        # Aggregate: sum messages arriving at each atom i
        agg = jnp.zeros((n_atoms, self.n_features)).at[idx_i].add(msgs)

        # Atom-wise update with residual connection
        delta = nn.Dense(self.n_features)(nn.silu(nn.Dense(self.n_features)(agg)))
        return h + delta                          # (n_atoms, F)


class SchNetPotential(nn.Module):
    '''
    Minimal SchNet graph neural network for molecular potential energy.

    Energy is invariant to rotations, translations, and permutations.
    Forces: F_i = -dE/dR_i  (computed via automatic differentiation).

    Attributes
    ----------
    max_atoms     : int   — fixed padded system size (4 for this dataset)
    n_features    : int   — embedding / hidden dimension
    n_interactions: int   — number of message-passing layers
    n_rbf         : int   — number of Gaussian RBF centers
    r_cut         : float — interaction cutoff in Bohr
    '''
    n_features:     int   = 32
    n_interactions: int   = 3
    n_rbf:          int   = 20
    r_cut:          float = 6.0

    @nn.compact
    def __call__(self, positions, real_mask = None):
        '''
        Parameters
        ----------
        positions : (max_atoms, 3)  — nuclear positions in Bohr (padded)
        real_mask : (max_atoms,)    — 1.0 for real H atoms, 0.0 for ghosts

        Returns
        -------
        energy : float  — total interaction energy in Hartree
        '''
        assert positions.ndim == 2 and positions.shape[1] == 3
        max_atoms = positions.shape[0]

        # All pairs (i,j) with i != j  [static, shape (n_pairs,)]
        pairs = [(i, j) for i in range(max_atoms)
                         for j in range(max_atoms) if i != j]
        idx_i = jnp.array([p[0] for p in pairs])   # (n_pairs,)
        idx_j = jnp.array([p[1] for p in pairs])

        # Pairwise distances (ghost atoms are far away → cutoff zeros them)
        r_ij = jnp.linalg.norm(
            positions[idx_i] - positions[idx_j], axis=-1
        )  # (n_pairs,)

        # RBF expansion with smooth cutoff
        rbf_ij = RBFLayer(n_rbf=self.n_rbf, r_cut=self.r_cut)(r_ij)

        # Initial atom embeddings (all H: type 0, single learned vector)
        h = nn.Embed(num_embeddings=1, features=self.n_features)(
            jnp.zeros(max_atoms, dtype=jnp.int32)
        )  # (max_atoms, n_features)

        # Message-passing layers
        for _ in range(self.n_interactions):
            h = InteractionBlock(n_features=self.n_features)(
                h, rbf_ij, idx_i, idx_j
            )

        # Per-atom readout → scalar energy per atom
        e = nn.Dense(self.n_features // 2)(h)
        e = nn.silu(e)
        e = nn.Dense(1)(e)[..., 0]               # (max_atoms,)

        # Sum only real atoms (ghosts are masked out)
        if real_mask is not None:
            return jnp.sum(e * real_mask)
        else:
            return jnp.sum(e)