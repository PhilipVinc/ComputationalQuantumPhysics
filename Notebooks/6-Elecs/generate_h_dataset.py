#!/usr/bin/env python3
"""
Generate h_dataset.npz — H2/H3/H4 potential energy surface dataset.
Uses PySCF RHF/STO-3G to compute energies and forces.
Expected runtime: 10-20 minutes on a laptop CPU.

Usage:
    python generate_h_dataset.py
"""

import numpy as np
from pyscf import gto, scf
import time, os

# -------------------------------------------------------------------
SEED       = 42
N_H2       = 800   # number of H2 configurations (2 electrons → RHF)
N_H3       = 0     # H3 has 3 electrons (odd) → skip to keep all-RHF
N_H4       = 400   # number of H4 configurations (4 electrons → RHF)
MAX_ATOMS  = 4     # pad all configs to this many atoms
# Ghost atom positions — must be DISTINCT so no two are coincident (avoids r=0 NaN)
GHOST_POSITIONS = np.array([
    [1000., 0., 0.],
    [0., 1000., 0.],
    [0., 0., 1000.],
], dtype=np.float64)  # up to 3 ghost atoms
OUTPUT     = 'h_dataset.npz'
# -------------------------------------------------------------------

E_H_ISOLATED = None  # computed below


def compute_pyscf(positions, basis='sto-3g'):
    """Return (energy, forces) for H atoms at given positions (Bohr)."""
    atom_str = '; '.join(f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in positions)
    mol = gto.Mole()
    mol.atom   = atom_str
    mol.basis  = basis
    mol.unit   = 'Bohr'
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 200
    e_tot = mf.kernel()
    if not mf.converged:
        return None, None
    g = mf.nuc_grad_method()
    g.verbose = 0
    forces = -g.kernel()   # F = -dE/dR
    return e_tot, forces


# ── Generate configurations ─────────────────────────────────────────

def gen_h2(n, r_min=0.5, r_max=5.5, rng=None):
    configs = []
    while len(configs) < n:
        r = rng.uniform(r_min, r_max)
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        configs.append(np.array([[0., 0., 0.], d * r]))
    return configs


def gen_h3(n, box=2.5, dmin=0.8, dmax=5.0, rng=None):
    configs, attempts = [], 0
    while len(configs) < n and attempts < 500_000:
        pos = rng.uniform(-box, box, size=(3, 3))
        dists = [np.linalg.norm(pos[i]-pos[j]) for i in range(3) for j in range(i+1,3)]
        if min(dists) >= dmin and max(dists) <= dmax:
            configs.append(pos)
        attempts += 1
    if len(configs) < n:
        print(f"  Warning: only generated {len(configs)}/{n} H3 configs")
    return configs


def gen_h4(n, box=3.0, dmin=0.8, dmax=6.0, rng=None):
    configs, attempts = [], 0
    while len(configs) < n and attempts < 500_000:
        pos = rng.uniform(-box, box, size=(4, 3))
        dists = [np.linalg.norm(pos[i]-pos[j]) for i in range(4) for j in range(i+1,4)]
        if min(dists) >= dmin and max(dists) <= dmax:
            configs.append(pos)
        attempts += 1
    if len(configs) < n:
        print(f"  Warning: only generated {len(configs)}/{n} H4 configs")
    return configs


# ── Main ─────────────────────────────────────────────────────────────

def main():
    global E_H_ISOLATED
    rng = np.random.default_rng(SEED)

    # Reference energy: RHF on well-separated H2 at 20 Bohr ≈ 2 * E(H)
    # (Single H has odd electrons; avoid open-shell by using widely-spaced H2)
    print("Computing isolated H atom energy (from H2 at 20 Bohr)...")
    pos_far = np.array([[0., 0., -10.], [0., 0., 10.]])
    e_h2_far, _ = compute_pyscf(pos_far)
    if e_h2_far is None:
        raise RuntimeError("Failed to compute reference energy")
    E_H_ISOLATED = e_h2_far / 2.0   # ≈ E(H_isolated) via RHF
    print(f"  E(H) ≈ {E_H_ISOLATED:.8f} Ha  (half of well-separated H2)")

    all_pos, all_E, all_F, all_n = [], [], [], []

    configs_h2 = gen_h2(N_H2, rng=rng) if N_H2 > 0 else []
    configs_h4 = gen_h4(N_H4, rng=rng) if N_H4 > 0 else []

    for label, n_atoms, configs in [
        ('H2', 2, configs_h2),
        ('H4', 4, configs_h4),
    ]:
        print(f"\nGenerating {len(configs)} {label} configurations...")
        n_ok, n_fail = 0, 0
        t0 = time.time()
        for i, pos in enumerate(configs):
            E, F = compute_pyscf(pos)
            if E is None:
                n_fail += 1
                continue
            # Pad to MAX_ATOMS — ghost atoms at DISTINCT far-away positions
            n_ghosts = MAX_ATOMS - n_atoms
            pos_pad  = np.vstack([pos, GHOST_POSITIONS[:n_ghosts]])
            F_pad    = np.zeros((MAX_ATOMS, 3))
            F_pad[:n_atoms] = F

            all_pos.append(pos_pad)
            all_E.append(E - n_atoms * E_H_ISOLATED)  # interaction energy
            all_F.append(F_pad)
            all_n.append(n_atoms)
            n_ok += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = n_ok / elapsed
                eta = (len(configs) - i - 1) / max(rate, 1e-9)
                print(f"  {label}: {i+1}/{len(configs)}  ok={n_ok} fail={n_fail}"
                      f"  rate={rate:.1f}/s  ETA={eta:.0f}s")

        print(f"  {label} done: {n_ok} ok, {n_fail} failed")

    positions = np.array(all_pos, dtype=np.float64)   # (N, 4, 3)
    energies  = np.array(all_E,  dtype=np.float64)    # (N,)
    forces    = np.array(all_F,  dtype=np.float64)    # (N, 4, 3)
    n_atoms   = np.array(all_n,  dtype=np.int32)      # (N,)

    np.savez(OUTPUT,
             positions=positions,
             energies=energies,
             forces=forces,
             n_atoms=n_atoms,
             e_h_isolated=np.array(E_H_ISOLATED))

    print(f"\nSaved {len(energies)} configurations to {OUTPUT}")
    print(f"  H2: {(n_atoms==2).sum()},  H4: {(n_atoms==4).sum()}")
    print(f"  Energy range: [{energies.min()*27211:.0f}, {energies.max()*27211:.0f}] meV")


if __name__ == '__main__':
    main()
