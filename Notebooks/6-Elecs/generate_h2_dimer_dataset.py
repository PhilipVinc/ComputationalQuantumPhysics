#!/usr/bin/env python3
"""
Generate H₂–H₂ dimer scan dataset for intermolecular potential training.

The existing H4 dataset (gen_h4 with box=3.0 Bohr) only samples compact
4-atom configurations with all atoms within ~3.2 Å.  It never samples
H₂-dimer geometries at 3–6 Å separation, so GNN models trained on it
cannot predict the intermolecular interaction potential.

This script fills that gap by scanning the COM–COM distance R from 2.0
to 6.5 Å in three orientations and four H–H bond lengths.

Geometry:
  Parallel  (‖) : both H₂ along z, COMs separated along x
  T-shape   (⊥) : mol-1 along z, mol-2 along y; COMs along x
  Tilted 45°(×) : mol-2 axis at 45° in xz-plane; COM along x

Output: dimer_dataset.npz — same format as h_dataset.npz:
  positions:    (N, 4, 3) float64  [Bohr]
  energies:     (N,)      float64  [Ha, relative to 4 × e_h_isolated]
  forces:       (N, 4, 3) float64  [Ha/Bohr]
  n_atoms:      (N,)      int32    [= 4 for all entries]
  e_h_isolated: scalar    float64  [copied from h_dataset.npz]

Expected runtime: ~10–15 min on a laptop CPU.
Usage:
    python generate_h2_dimer_dataset.py
"""

import numpy as np
from pyscf import gto, scf
import time

BOHR_TO_ANG = 0.529177

# ── Reference energy ──────────────────────────────────────────────────
data = np.load('h_dataset.npz')
E_H_ISOLATED = float(data['e_h_isolated'])
print(f"E_H_ISOLATED = {E_H_ISOLATED:.8f} Ha  (from h_dataset.npz)")


def compute_pyscf(positions_bohr):
    """RHF/STO-3G energy and forces for 4 H atoms.

    Parameters
    ----------
    positions_bohr : (4, 3) array, atom positions in Bohr.

    Returns
    -------
    (e_tot, forces) : (float, (4,3) array)  Ha and Ha/Bohr, or (None, None).
    """
    atom_str = '; '.join(f'H {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}'
                         for r in positions_bohr)
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = 'sto-3g'
    mol.unit = 'Bohr'
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        return None, None
    g = mf.nuc_grad_method()
    g.verbose = 0
    forces = -g.kernel()     # F = −dE/dR, shape (4, 3)
    return mf.e_tot, forces


# ── Build geometry list ───────────────────────────────────────────────
R_ang_values = np.linspace(2.0, 6.5, 25)   # COM–COM distance in Å
d_ang_values = [0.60, 0.70, 0.80, 0.90]    # H–H bond length in Å

configs = []   # list of (4, 3) position arrays in Bohr

for d_ang in d_ang_values:
    d = d_ang / BOHR_TO_ANG   # Bohr
    for R_ang in R_ang_values:
        R = R_ang / BOHR_TO_ANG   # Bohr

        # 1. Parallel: both H₂ along z, COMs separated along x
        configs.append(np.array([
            [0.,  0., -d/2], [0.,  0.,  d/2],
            [R,   0., -d/2], [R,   0.,  d/2],
        ], dtype=np.float64))

        # 2. T-shape: mol-1 along z, mol-2 along y; COMs along x
        configs.append(np.array([
            [0.,   0., -d/2], [0.,   0.,  d/2],
            [R,  -d/2,   0.], [R,   d/2,  0.],
        ], dtype=np.float64))

        # 3. Tilted 45°: mol-2 axis at 45° in xz-plane, COM along x
        dh = d / (2.0 * np.sqrt(2.0))
        configs.append(np.array([
            [0.,    0., -d/2], [0.,    0.,  d/2],
            [R-dh,  0., -dh],  [R+dh,  0.,  dh],
        ], dtype=np.float64))

n_total = len(configs)
print(f"Total configurations: {n_total}  "
      f"({len(d_ang_values)} d-values × {len(R_ang_values)} R-values × 3 orientations)")

# ── Compute PySCF ─────────────────────────────────────────────────────
all_pos, all_E, all_F = [], [], []
n_ok, n_fail = 0, 0
t0 = time.time()

for k, pos in enumerate(configs):
    E_tot, forces = compute_pyscf(pos)

    if E_tot is None:
        n_fail += 1
        if k % 50 == 0:
            print(f"  {k}/{n_total}  WARN: SCF did not converge, skipping")
        continue

    all_pos.append(pos)
    all_E.append(E_tot - 4.0 * E_H_ISOLATED)
    all_F.append(forces)
    n_ok += 1

    if (k + 1) % 50 == 0:
        elapsed = time.time() - t0
        eta = (n_total - k - 1) / max(n_ok / elapsed, 1e-9)
        print(f"  {k+1}/{n_total}  ok={n_ok}  fail={n_fail}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

# ── Save ──────────────────────────────────────────────────────────────
all_pos = np.array(all_pos, dtype=np.float64)    # (N, 4, 3) Bohr
all_E   = np.array(all_E,   dtype=np.float64)    # (N,)      Ha
all_F   = np.array(all_F,   dtype=np.float64)    # (N, 4, 3) Ha/Bohr
n_atoms = np.full(n_ok, 4, dtype=np.int32)

np.savez('dimer_dataset.npz',
         positions    = all_pos,
         energies     = all_E,
         forces       = all_F,
         n_atoms      = n_atoms,
         e_h_isolated = np.float64(E_H_ISOLATED))

EV = 27211.4
print(f"\nSaved dimer_dataset.npz  ({n_ok} configs, {n_fail} failed)")
print(f"  Energy range: [{all_E.min()*EV:.0f}, {all_E.max()*EV:.0f}] meV")
