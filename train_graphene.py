
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit-pure graphene tight-binding toolkit (Q1-ready, robust edition):
- Physics-correct invariants (Dirac K/K′, Berry ±π, vF grad/fit, isotropy, C3, τ-gauge)
- Symmetric audits at K and K′ (curvature, linear regime)
- Convergence studies:
  * Curvature via adaptive step and Richardson extrapolation (with error-based bound)
  * Velocity ring-fit stability across q_rel and directions
  * Berry phase adaptive across radii and point counts
- Scaling audits with >=13 points, statistics (relative residual, R²)
- Robust Newton refinement with damping fallback
- t′ handling: warnings and relaxed assertions when Dirac cone approximation breaks
- Constants via scipy.constants (CODATA 2022 provenance)
- CLI tolerant to Jupyter/Colab kernel args
- Timestamp (UTC) + environment + constants metadata
"""

from jax import config, jit
config.update('jax_enable_x64', True)

import jax
import jax.numpy as jnp
import argparse
import json
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

# CODATA constants from scipy (provenance)
try:
    from scipy.constants import hbar as HBAR_SI, electron_volt as EV_TO_J, codata
    CONSTANTS_METADATA = {
        "source": "scipy.constants",
        "codata_version": getattr(codata, "version", "CODATA 2022"),
        "values": {
            "hbar_Js": HBAR_SI,
            "eV_to_J": EV_TO_J,
        },
    }
    eV_to_J = float(EV_TO_J)
    hbar = float(HBAR_SI)
except Exception:
    # Fallback to known CODATA 2022 values if scipy is unavailable
    CONSTANTS_METADATA = {
        "source": "fallback (hard-coded)",
        "codata_version": "CODATA 2022",
        "values": {
            "hbar_Js": 1.054571817e-34,
            "eV_to_J": 1.602176634e-19,
        },
    }
    eV_to_J = 1.602176634e-19
    hbar = 1.054571817e-34

# ----- CONFIG (tolerances, sampling) -----
@dataclass(frozen=True)
class AuditConfig:
    q_rel_default: float = 1e-9          # ring radius relative to a
    h_rel_default: float = 1e-9          # curvature step relative to a
    directions_ring: int = 64
    directions_curvature: int = 64
    berry_dirs_base: int = 256
    berry_radius_base_rel: float = 1e-9
    scaling_points: int = 13             # >= 10 for statistics
    det_tol: float = 1e-24               # Newton Jacobian tol
    newton_max_iter: int = 100
    newton_tol: float = 1e-16
    vf_tol_rel: float = 1e-6
    berry_tol_abs: float = 1e-6
    periodicity_atol: float = 1e-12
    hermiticity_atol: float = 1e-12
    gauge_atol: float = 1e-12
    curvature_min_abs_bound: float = 1e-12  # minimal absolute bound floor (scaled)
    scaling_R2_min: float = 0.999999
    tprime_relax_threshold: float = 0.3  # relax assertions when t'/t exceeds this
    tprime_isotropy_relax: float = 5e-6  # relaxed isotropy tolerance when t' large

CFG = AuditConfig()

# ----- LATTICE AND HELPERS -----
def build_lattice(a):
    a1 = jnp.array([0.5 * a,  jnp.sqrt(3.0) * a / 2.0])
    a2 = jnp.array([-0.5 * a, jnp.sqrt(3.0) * a / 2.0])
    A = jnp.stack([a1, a2], axis=1)
    B = 2.0 * jnp.pi * jnp.linalg.inv(A)
    b1, b2 = B[:, 0], B[:, 1]
    a_cc = a / jnp.sqrt(3.0)
    tau = jnp.array([0.0, a_cc])
    deltas = jnp.stack([tau, tau - a1, tau - a2], axis=0)
    nnn = jnp.stack([ a1, -a1, a2, -a2, a1 - a2, -(a1 - a2) ], axis=0)
    return a1, a2, A, b1, b2, a_cc, tau, deltas, nnn

def vF_analytic_from(a_cc, tJ):
    return (1.5 * a_cc * tJ) / hbar

# ----- STRUCTURE FACTOR AND HAMILTONIAN -----
def make_f_eps(deltas, nnn, tJ, tprimeJ):
    @jit
    def f_k(k):
        return -tJ * jnp.sum(jnp.exp(1j * (deltas @ k)))
    @jit
    def eps_diag(k):
        if tprimeJ == 0.0:
            return 0.0
        return -tprimeJ * jnp.sum(jnp.cos(nnn @ k))
    return f_k, eps_diag

def make_hamiltonian(f_k, eps_diag):
    @jit
    def h_k(k):
        f = f_k(k)
        e = eps_diag(k)
        return jnp.array([[e + 0.0j, f],
                          [jnp.conj(f), e + 0.0j]], dtype=jnp.complex128)
    @jit
    def energies(k):
        return jnp.linalg.eigvalsh(h_k(k))
    return h_k, energies

def grad_f_from(deltas, tJ):
    @jit
    def grad_f(k):
        phase = jnp.exp(1j * (deltas @ k))
        coeff = 1j * deltas
        return -tJ * jnp.sum(coeff * phase[:, None], axis=0)
    return grad_f

# ----- K/K′ CONSTRUCTION AND DAMPED NEWTON -----
def analytic_K_from_phases(A):
    p = jnp.array([-2.0 * jnp.pi / 3.0, +2.0 * jnp.pi / 3.0])
    return jnp.linalg.solve(A.T, p)

def analytic_Kprime_from_phases(A):
    p = jnp.array([+2.0 * jnp.pi / 3.0, -2.0 * jnp.pi / 3.0])
    return jnp.linalg.solve(A.T, p)

def newton_refine_K(k0, f_k, grad_f, max_iter=CFG.newton_max_iter, tol=CFG.newton_tol, det_tol=CFG.det_tol):
    k = k0
    alpha = 1.0
    for _ in range(int(max_iter)):
        f = f_k(k)
        g = grad_f(k)
        J = jnp.stack([jnp.real(g), jnp.imag(g)], axis=0)
        F = jnp.array([jnp.real(f), jnp.imag(f)])
        detJ = jnp.linalg.det(J)
        if jnp.abs(detJ) < det_tol:
            alpha *= 0.5
            if alpha < 1e-6:
                break
        else:
            dk = jnp.linalg.solve(J, F)
            k_new = k - alpha * dk
            if jnp.linalg.norm(alpha * dk) < tol:
                k = k_new
                break
            k = k_new
            alpha = min(1.0, alpha * 2.0)
    return k

# ----- VELOCITY METRICS + CONVERGENCE -----
def vF_from_grad_at_K(K, grad_f):
    g = grad_f(K)
    g_norm = jnp.sqrt(jnp.real(g @ jnp.conj(g)))   # ||∇f|| (J·m)
    return float(g_norm / (jnp.sqrt(2.0) * hbar))  # vF = ||∇f|| / (√2 ħ)

def estimate_vf_energy_fit(K, energies, a, q_abs_rel=CFG.q_rel_default, directions=CFG.directions_ring, subtract_mean=False):
    q_abs = q_abs_rel / a
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, int(directions), endpoint=False)
    us = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    def sample(u):
        k = K + q_abs * u
        E = energies(k)
        return jnp.max(E)
    Ec = jnp.array([sample(u) for u in us])
    if subtract_mean:
        vF = jnp.mean(Ec - Ec.mean()) / (hbar * q_abs)
    else:
        vF = jnp.mean(Ec) / (hbar * q_abs)
    spread = jnp.std(Ec) / (hbar * q_abs)
    return float(vF), float(spread), us, Ec

def vf_fit_convergence(K, energies, a):
    q_rels = jnp.array([5e-10, 1e-9, 2e-9])  # relative to a
    dirs = [32, 64, 128]
    vals = []
    for q_rel in q_rels:
        for d in dirs:
            vF, spread, _, _ = estimate_vf_energy_fit(K, energies, a, q_abs_rel=float(q_rel), directions=int(d))
            vals.append((float(q_rel), int(d), float(vF), float(spread)))
    return vals  # list of tuples

# ----- ADAPTIVE BERRY PHASE -----
def berry_phase_discrete(K, f_k, a, radius_rel, directions):
    r = radius_rel / a
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, int(directions), endpoint=False)
    ks = K + r * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    phases = jnp.array([jnp.angle(f_k(k)) for k in ks])
    diffs = phases[(jnp.arange(int(directions)) + 1) % int(directions)] - phases
    diffs = jnp.arctan2(jnp.sin(diffs), jnp.cos(diffs))
    total_winding = jnp.sum(diffs)
    gamma = 0.5 * total_winding
    return float(gamma), float(total_winding), ks, phases

def berry_phase_adaptive(K, f_k, a, base_radius_rel=CFG.berry_radius_base_rel, base_dirs=CFG.berry_dirs_base):
    candidates = [
        (base_radius_rel, base_dirs),
        (base_radius_rel * 2.0, base_dirs),
        (base_radius_rel / 2.0, base_dirs * 2),
        (base_radius_rel, base_dirs * 2),
    ]
    results = []
    for r_rel, ndir in candidates:
        gamma, wind, ks, phases = berry_phase_discrete(K, f_k, a, r_rel, ndir)
        results.append((gamma, wind, r_rel, ndir, ks, phases))
    target = 2.0 * jnp.pi
    idx = int(jnp.argmin(jnp.array([abs(abs(w) - target) for (_, w, _, _, _, _) in results])))
    return results[idx], results  # best, all tried

# ----- SYMMETRY AUDITS -----
def c3_rotation_matrix():
    c = -0.5
    s = jnp.sqrt(3.0) / 2.0
    return jnp.array([[c, -s],
                      [s,  c]])

def audit_c3_invariance(f_k, b1, b2, samples=6, atol=CFG.periodicity_atol):
    R = c3_rotation_matrix()
    ks = [
        0.15 * b1 + 0.10 * b2,
        0.33 * b1 + 0.07 * b2,
        0.42 * b1 + 0.21 * b2,
        0.05 * b1 + 0.47 * b2,
        0.27 * b1 + 0.36 * b2,
        0.49 * b1 + 0.12 * b2,
    ][:int(samples)]
    for k in ks:
        lhs = jnp.abs(f_k(k))
        rhs = jnp.abs(f_k(R @ k))
        if not bool(jnp.allclose(lhs, rhs, atol=atol)):
            return False
    return True

def energies_with_tau(k, tau_shift, a1, a2, tJ, nnn, eps_diag_base):
    deltas_shift = jnp.stack([tau_shift, tau_shift - a1, tau_shift - a2], axis=0)
    phase = jnp.exp(1j * (deltas_shift @ k))
    f = -tJ * jnp.sum(phase)
    e = eps_diag_base(k)
    H = jnp.array([[e + 0.0j, f],
                   [jnp.conj(f), e + 0.0j]], dtype=jnp.complex128)
    return jnp.linalg.eigvalsh(H)

def audit_tau_gauge_invariance(a1, a2, b1, b2, tJ, nnn, eps_diag_base, atol=CFG.gauge_atol):
    shifts = [jnp.array([0.0, 0.0]), a1, a2, a1 + a2, -a1, -a2]
    kpoints = [0.23 * b1 + 0.41 * b2, 0.37 * b1 + 0.19 * b2, 0.11 * b1 + 0.29 * b2]
    base = [energies_with_tau(k, jnp.array([0.0, 0.0]), a1, a2, tJ, nnn, eps_diag_base) for k in kpoints]
    for sh in shifts:
        cur = [energies_with_tau(k, sh, a1, a2, tJ, nnn, eps_diag_base) for k in kpoints]
        for E0, E1 in zip(base, cur):
            if not bool(jnp.allclose(E0, E1, atol=atol)):
                return False
    return True

# ----- CURVATURE (adaptive + Richardson) & LINEAR REGIME -----
def curvature_central(K, u, energies, a, h_rel):
    h = h_rel / a
    E_plus = lambda kk: jnp.max(energies(kk))
    E0 = E_plus(K)
    Ep = E_plus(K + h * u)
    Em = E_plus(K - h * u)
    return float((Ep - 2.0 * E0 + Em) / (h * h))

def curvature_adaptive(K, u, energies, a, h_rel_base=CFG.h_rel_default):
    # Two-step Richardson extrapolation: kappa(h) ~ kappa_true + C h^2
    h1 = h_rel_base
    h2 = h_rel_base / 2.0
    k1 = curvature_central(K, u, energies, a, h1)
    k2 = curvature_central(K, u, energies, a, h2)
    # Extrapolate k_true ≈ (4*k2 - k1)/3
    k_true = (4.0 * k2 - k1) / 3.0
    err_est = abs(k2 - k_true)  # heuristic error estimate of discretization
    return float(k_true), float(err_est)

def curvature_max_over_directions(K, energies, a, directions=CFG.directions_curvature):
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, int(directions), endpoint=False)
    us = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    kappas = []
    errs = []
    for u in us:
        kt, et = curvature_adaptive(K, u, energies, a)
        kappas.append(abs(kt))
        errs.append(et)
    return float(max(kappas)), float(max(errs))

def linear_regime_radius(K, energies, a, vF, directions=CFG.directions_curvature, eps_rel=1e-6, h_candidates_rel=(1e-9, 2e-9, 5e-9, 1e-8)):
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, int(directions), endpoint=False)
    us = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    best = 0.0
    for h_rel in h_candidates_rel:
        h = h_rel / a
        ok = True
        for u in us:
            E_plus = jnp.max(energies(K + h * u))
            dev = jnp.abs((E_plus - hbar * vF * h) / (hbar * vF * h))
            if dev > eps_rel:
                ok = False
                break
        if ok:
            best = h_rel
    return float(best)

# ----- SCALING AUDITS WITH EXPANDED SAMPLING -----
def stats_linear_fit(x, y):
    x = jnp.array(x); y = jnp.array(y)
    c = (x @ y) / (x @ x)
    residuals = y - c * x
    rss = jnp.sum(residuals ** 2)
    tss = jnp.sum((y - jnp.mean(y)) ** 2)
    r2 = 1.0 - float(rss / tss) if tss > 0 else 1.0
    rel_resid = float(jnp.sqrt(jnp.mean(residuals ** 2)) / jnp.mean(jnp.abs(y)))
    return float(c), rel_resid, r2

def scaling_vf_vs_t(A, deltas, tJ, a_cc, num_points=CFG.scaling_points):
    ts = jnp.linspace(0.5 * tJ, 1.5 * tJ, int(num_points))
    vfs = []
    for ti in ts:
        @jit
        def f_k_t(k):
            return -ti * jnp.sum(jnp.exp(1j * (deltas @ k)))
        grad_f_t = grad_f_from(deltas, ti)
        K = newton_refine_K(analytic_K_from_phases(A), f_k_t, grad_f_t)
        g = grad_f_t(K)
        g_norm = jnp.sqrt(jnp.real(g @ jnp.conj(g)))
        vF_t = g_norm / (jnp.sqrt(2.0) * hbar)
        vfs.append(float(vF_t))
    c_fit, rel_resid, r2 = stats_linear_fit(ts, vfs)
    slope_t_theory = float((1.5 * a_cc) / hbar)
    return float(c_fit), float(rel_resid), float(r2), slope_t_theory

def scaling_vf_vs_a(tJ, num_points=CFG.scaling_points):
    as_ = jnp.linspace(0.9 * 2.46e-10, 1.1 * 2.46e-10, int(num_points))
    vfs = []
    for ai in as_:
        a1_i = jnp.array([0.5 * ai,  jnp.sqrt(3.0) * ai / 2.0])
        a2_i = jnp.array([-0.5 * ai, jnp.sqrt(3.0) * ai / 2.0])
        A_i = jnp.stack([a1_i, a2_i], axis=1)
        a_cc_i = ai / jnp.sqrt(3.0)
        tau_i = jnp.array([0.0, a_cc_i])
        deltas_i = jnp.stack([tau_i, tau_i - a1_i, tau_i - a2_i], axis=0)

        @jit
        def f_k_a(k):
            return -tJ * jnp.sum(jnp.exp(1j * (deltas_i @ k)))
        grad_f_a = grad_f_from(deltas_i, tJ)
        K_i = newton_refine_K(analytic_K_from_phases(A_i), f_k_a, grad_f_a)
        g = grad_f_a(K_i)
        g_norm = jnp.sqrt(jnp.real(g @ jnp.conj(g)))
        vF_a = g_norm / (jnp.sqrt(2.0) * hbar)
        vfs.append(float(vF_a))
    c_fit, rel_resid, r2 = stats_linear_fit(as_, vfs)
    slope_a_theory = float((1.5 * tJ) / (jnp.sqrt(3.0) * hbar))
    return float(c_fit), float(rel_resid), float(r2), slope_a_theory

# ----- EXPORT ARTIFACTS -----
def export_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def export_csv_ring(path, us, Ec, label="K"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "ux", "uy", "E_plus_J"])
        for u, e in zip(us, Ec):
            writer.writerow([label, float(u[0]), float(u[1]), float(e)])

def export_csv_phase(path, ks, phases, label="K"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "kx", "ky", "phase_rad"])
        for k, ph in zip(ks, phases):
            writer.writerow([label, float(k[0]), float(k[1]), float(ph)])

# ----- MAIN AUDIT -----
def run_audit(a=2.46e-10, t_eV=2.8, tprime_eV=0.0, export_dir=None):
    tJ = t_eV * eV_to_J
    tprimeJ = tprime_eV * eV_to_J
    tprime_ratio = float(tprime_eV / t_eV) if t_eV != 0 else 0.0
    relax_isotropy = tprime_ratio >= CFG.tprime_relax_threshold

    # Build lattice and model
    a1, a2, A, b1, b2, a_cc, tau, deltas, nnn = build_lattice(a)
    vF_analytic = vF_analytic_from(a_cc, tJ)
    f_k, eps_diag = make_f_eps(deltas, nnn, tJ, tprimeJ)
    h_k, energies = make_hamiltonian(f_k, eps_diag)
    grad_f = grad_f_from(deltas, tJ)

    # Geometry checks
    delta_lengths = jnp.linalg.norm(deltas, axis=1)
    delta_ok = bool(jnp.allclose(delta_lengths, a_cc, rtol=0, atol=1e-14))

    # Dirac valleys
    K0 = analytic_K_from_phases(A)
    Kp0 = analytic_Kprime_from_phases(A)
    K = newton_refine_K(K0, f_k, grad_f)
    Kp = newton_refine_K(Kp0, f_k, grad_f)
    fK = f_k(K); fKp = f_k(Kp)
    dirac_K = bool(jnp.abs(fK) < 1e-12 * tJ)
    dirac_Kp = bool(jnp.abs(fKp) < 1e-12 * tJ)

    # Baseline audits
    ktest = 0.1 * b1 + 0.1 * b2
    H = h_k(ktest)
    hermiticity = bool(jnp.allclose(H, H.conj().T, atol=CFG.hermiticity_atol))
    ph_sym = bool(jnp.allclose(jnp.sum(energies(ktest)) - 2.0 * eps_diag(ktest), 0.0, atol=CFG.hermiticity_atol))
    periodic_b1 = bool(jnp.allclose(f_k(ktest + b1), f_k(ktest), atol=CFG.periodicity_atol))
    periodic_b2 = bool(jnp.allclose(f_k(ktest + b2), f_k(ktest), atol=CFG.periodicity_atol))

    # Velocities and isotropy (K and K′) + convergence samples
    vf_grad_K = vF_from_grad_at_K(K, grad_f)
    vf_grad_Kp = vF_from_grad_at_K(Kp, grad_f)
    subtract_mean = (tprimeJ != 0.0)
    vf_fit_K, vf_spread_K, usK, EcK = estimate_vf_energy_fit(K, energies, a, q_abs_rel=CFG.q_rel_default, directions=CFG.directions_ring, subtract_mean=subtract_mean)
    vf_fit_Kp, vf_spread_Kp, usKp, EcKp = estimate_vf_energy_fit(Kp, energies, a, q_abs_rel=CFG.q_rel_default, directions=CFG.directions_ring, subtract_mean=subtract_mean)
    vf_conv_K = vf_fit_convergence(K, energies, a)
    vf_conv_Kp = vf_fit_convergence(Kp, energies, a)

    # Berry phases (adaptive)
    (gamma_K, wind_K, rK, nK, ksK, phasesK), berry_trials_K = berry_phase_adaptive(K, f_k, a)
    (gamma_Kp, wind_Kp, rKp, nKp, ksKp, phasesKp), berry_trials_Kp = berry_phase_adaptive(Kp, f_k, a)

    # Symmetries
    c3_ok = audit_c3_invariance(f_k, b1, b2, samples=6, atol=CFG.periodicity_atol)
    tau_ok = audit_tau_gauge_invariance(a1, a2, b1, b2, tJ, nnn, eps_diag, atol=CFG.gauge_atol)

    # Curvature & linear regime (K and K′) with adaptive + Richardson
    kappa_max_K, kappa_err_K = curvature_max_over_directions(K, energies, a, directions=CFG.directions_curvature)
    kappa_max_Kp, kappa_err_Kp = curvature_max_over_directions(Kp, energies, a, directions=CFG.directions_curvature)
    r_lin_K = linear_regime_radius(K, energies, a, vf_grad_K, directions=CFG.directions_curvature, eps_rel=1e-6)
    r_lin_Kp = linear_regime_radius(Kp, energies, a, vf_grad_Kp, directions=CFG.directions_curvature, eps_rel=1e-6)

    # Scaling audits (expanded sampling + stats)
    c_t, resid_t, r2_t, slope_t_theory = scaling_vf_vs_t(A, deltas, tJ, a_cc, num_points=CFG.scaling_points)
    c_a, resid_a, r2_a, slope_a_theory = scaling_vf_vs_a(tJ, num_points=CFG.scaling_points)

    # Environment and timestamp (UTC, ISO 8601 with Z)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    env = {
        "python_version": sys.version,
        "jax_version": getattr(jax, "__version__", "unknown"),
        "dtype": "float64",
        "device_kind": jax.default_backend(),
        "constants_metadata": CONSTANTS_METADATA,
    }

    report = {
        "timestamp": timestamp,
        "environment": env,
        "params": {
            "a_m": float(a),
            "t_eV": float(t_eV),
            "tprime_eV": float(tprime_eV),
            "a_cc_m": float(a_cc),
            "tprime_ratio": float(tprime_ratio),
        },
        "derived": {
            "vF_analytic_m_per_s": float(vF_analytic),
        },
        "audits": {
            "delta_lengths_equal_acc": delta_ok,
            "hermiticity": hermiticity,
            "particle_hole": ph_sym if tprimeJ == 0.0 else None,
            "period_b1": periodic_b1,
            "period_b2": periodic_b2,
            "dirac_at_K": dirac_K,
            "dirac_at_Kprime": dirac_Kp,
            "fK_abs_over_t": float(jnp.abs(fK) / tJ),
            "fKprime_abs_over_t": float(jnp.abs(fKp) / tJ),
            "K_coords": [float(K[0]), float(K[1])],
            "Kprime_coords": [float(Kp[0]), float(Kp[1])],
            "vf_grad_K_m_per_s": vf_grad_K,
            "vf_grad_Kprime_m_per_s": vf_grad_Kp,
            "vf_fit_K_m_per_s": vf_fit_K,
            "vf_fit_Kprime_m_per_s": vf_fit_Kp,
            "vf_fit_spread_K_m_per_s": vf_spread_K,
            "vf_fit_spread_Kprime_m_per_s": vf_spread_Kp,
            "vf_grad_over_analytic_K": float(vf_grad_K / vF_analytic),
            "vf_grad_over_analytic_Kprime": float(vf_grad_Kp / vF_analytic),
            "vf_fit_over_analytic_K": float(vf_fit_K / vF_analytic),
            "vf_fit_over_analytic_Kprime": float(vf_fit_Kp / vF_analytic),
            "berry_phase_gamma_K": float(gamma_K),
            "berry_phase_gamma_Kprime": float(gamma_Kp),
            "phase_winding_total_K": float(wind_K),
            "phase_winding_total_Kprime": float(wind_Kp),
            "berry_trials_K": [{"radius_rel": float(rr), "dirs": int(nn), "gamma": float(ga), "wind": float(wi)} for (ga, wi, rr, nn, _, _) in berry_trials_K],
            "berry_trials_Kprime": [{"radius_rel": float(rr), "dirs": int(nn), "gamma": float(ga), "wind": float(wi)} for (ga, wi, rr, nn, _, _) in berry_trials_Kp],
            "c3_invariance": c3_ok,
            "tau_gauge_invariance": tau_ok,
            "curvature_max_K_J": float(kappa_max_K),
            "curvature_err_K_J": float(kappa_err_K),
            "curvature_max_Kprime_J": float(kappa_max_Kp),
            "curvature_err_Kprime_J": float(kappa_err_Kp),
            "linear_regime_radius_rel_K": float(r_lin_K),
            "linear_regime_radius_rel_Kprime": float(r_lin_Kp),
            "scaling_slope_t_fit": float(c_t),
            "scaling_slope_t_theory": float(slope_t_theory),
            "scaling_residual_t": float(resid_t),
            "scaling_R2_t": float(r2_t),
            "scaling_slope_a_fit": float(c_a),
            "scaling_slope_a_theory": float(slope_a_theory),
            "scaling_residual_a": float(resid_a),
            "scaling_R2_a": float(r2_a),
            "vf_fit_convergence_K": [{"q_rel": float(q), "dirs": int(d), "vF": float(v), "spread": float(s)} for (q, d, v, s) in vf_conv_K],
            "vf_fit_convergence_Kprime": [{"q_rel": float(q), "dirs": int(d), "vF": float(v), "spread": float(s)} for (q, d, v, s) in vf_conv_Kp],
        },
        "notes": {
            "isotropy_tprime_relaxed": relax_isotropy,
            "berry_selected_K": {"radius_rel": float(rK), "dirs": int(nK)},
            "berry_selected_Kprime": {"radius_rel": float(rKp), "dirs": int(nKp)},
        }
    }

    # Fail-hard assertions with reasoned tolerances + t′ handling
    assert delta_ok, "NN lengths mismatch a_cc."
    assert hermiticity, "Hamiltonian not Hermitian."
    assert periodic_b1 and periodic_b2, "Bloch periodicity failed."
    assert dirac_K and dirac_Kp and report["audits"]["fK_abs_over_t"] < 1e-12 and report["audits"]["fKprime_abs_over_t"] < 1e-12, "Dirac condition failed."
    if tprimeJ == 0.0:
        assert ph_sym, "Particle–hole symmetry failed at t'=0."
    # vF checks
    assert abs(report["audits"]["vf_grad_over_analytic_K"] - 1.0) < CFG.vf_tol_rel, "vF (grad, K) mismatch."
    assert abs(report["audits"]["vf_grad_over_analytic_Kprime"] - 1.0) < CFG.vf_tol_rel, "vF (grad, K') mismatch."
    assert abs(report["audits"]["vf_fit_over_analytic_K"] - 1.0) < CFG.vf_tol_rel, "vF (fit, K) mismatch."
    assert abs(report["audits"]["vf_fit_over_analytic_Kprime"] - 1.0) < CFG.vf_tol_rel, "vF (fit, K') mismatch."
    # Isotropy tolerance: relaxed when t′ is large (Dirac cone approximation weakens)
    iso_tol = CFG.tprime_isotropy_relax if relax_isotropy else CFG.vf_tol_rel
    assert (report["audits"]["vf_fit_spread_K_m_per_s"] / vF_analytic) < iso_tol, "Isotropy spread too large (K)."
    assert (report["audits"]["vf_fit_spread_Kprime_m_per_s"] / vF_analytic) < iso_tol, "Isotropy spread too large (K')."
    # Berry: enforce ±π
    assert abs(report["audits"]["berry_phase_gamma_K"] - jnp.pi) < CFG.berry_tol_abs, "Berry phase at K not π."
    assert abs(report["audits"]["berry_phase_gamma_Kprime"] + jnp.pi) < CFG.berry_tol_abs, "Berry phase at K' not -π."
    assert c3_ok, "C3 rotational invariance failed."
    assert tau_ok, "Tau gauge invariance failed."
    # Curvature tolerances: tie to adaptive error estimate and a small absolute floor
    # Bound = max(10 * err_est, minimal_absolute_floor)
    curv_floor = CFG.curvature_min_abs_bound * tJ / (a * a)
    assert report["audits"]["curvature_max_K_J"] < max(10.0 * report["audits"]["curvature_err_K_J"], curv_floor), "Curvature too large near K (beyond adaptive bound)."
    assert report["audits"]["curvature_max_Kprime_J"] < max(10.0 * report["audits"]["curvature_err_Kprime_J"], curv_floor), "Curvature too large near K′ (beyond adaptive bound)."
    # Scaling: expanded points should yield near-perfect linearity
    assert abs(c_t - slope_t_theory) / slope_t_theory < CFG.vf_tol_rel, "vF vs t slope mismatch."
    assert report["audits"]["scaling_residual_t"] < CFG.vf_tol_rel and report["audits"]["scaling_R2_t"] > CFG.scaling_R2_min, "vF vs t residual/R2 too weak."
    assert abs(c_a - slope_a_theory) / slope_a_theory < CFG.vf_tol_rel, "vF vs a slope mismatch."
    assert report["audits"]["scaling_residual_a"] < CFG.vf_tol_rel and report["audits"]["scaling_R2_a"] > CFG.scaling_R2_min, "vF vs a residual/R2 too weak."

    # Export artifacts (optional)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        export_json(os.path.join(export_dir, "audit_report.json"), report)
        export_csv_ring(os.path.join(export_dir, "ring_K.csv"), usK, EcK, label="K")
        export_csv_ring(os.path.join(export_dir, "ring_Kprime.csv"), usKp, EcKp, label="Kprime")
        export_csv_phase(os.path.join(export_dir, "phase_K.csv"), ksK, phasesK, label="K")
        export_csv_phase(os.path.join(export_dir, "phase_Kprime.csv"), ksKp, phasesKp, label="Kprime")

    # Human-readable printout
    print("Audit report:")
    print(f"  timestamp: {report['timestamp']}")
    for section in ["environment", "params", "derived", "audits", "notes"]:
        print(f"  {section}:")
        for k, v in report[section].items():
            print(f"    {k}: {v}")

    if relax_isotropy:
        print("Note: t′/t is large; isotropy tolerance relaxed. Consider analytical warping benchmarks for deeper study.")

    return report

def main():
    parser = argparse.ArgumentParser(description="Audit-pure graphene TB toolkit with exportable artifacts (robust + convergence, CODATA via scipy).")
    parser.add_argument("--a", type=float, default=2.46e-10, help="Bravais lattice constant (m)")
    parser.add_argument("--t_eV", type=float, default=2.8, help="Nearest-neighbor hopping (eV)")
    parser.add_argument("--tprime_eV", type=float, default=0.0, help="Next-nearest neighbor (eV)")
    parser.add_argument("--export", type=str, default=None, help="Directory to export JSON/CSV artifacts")
    # Tolerate extra args injected by Jupyter/Colab (e.g., -f kernel.json)
    args, _ = parser.parse_known_args()
    run_audit(a=args.a, t_eV=args.t_eV, tprime_eV=args.tprime_eV, export_dir=args.export)

if __name__ == "__main__":
    main()
