"""
Brute-force optimizer for second-order DC motor model parameters.
Finds R, L, J, B, kt, ke that minimize combined RMSE against measured data.
Prints the best values found — paste them into main.py.
"""
import numpy as np
import csv

# --- Load measured data ---
time_ms_list, vel_list = [], []
with open('data.csv', 'r') as f:
    for row in csv.DictReader(f):
        time_ms_list.append(int(row['time_ms']))
        vel_list.append(int(row['velocity']))

t_data = np.array(time_ms_list) / 1000.0
meas_vel = np.array(vel_list, dtype=float)

T_SWITCH = 1.0
V = 12.0

accel_mask = t_data <= T_SWITCH
decel_mask = ~accel_mask
t_acc = t_data[accel_mask]
v_acc = meas_vel[accel_mask]
t_dec = t_data[decel_mask] - T_SWITCH
v_dec = meas_vel[decel_mask]


# --- Model (copied from main.py) ---
def solve_phase(t, alpha, beta_sq, w_ss, w0, wd0):
    omega = np.empty_like(t)
    omega_dot = np.empty_like(t)

    if beta_sq > 1e-10:  # underdamped
        beta = np.sqrt(beta_sq)
        A = w0 - w_ss
        C = (wd0 - alpha * A) / beta
        eat = np.exp(np.clip(alpha * t, -500, 500))
        cb, sb = np.cos(beta * t), np.sin(beta * t)
        omega[:] = w_ss + eat * (A * cb + C * sb)
        omega_dot[:] = eat * ((alpha * A + beta * C) * cb
                              + (alpha * C - beta * A) * sb)
    elif beta_sq < -1e-10:  # overdamped
        gamma = np.sqrt(-beta_sq)
        s1, s2 = alpha + gamma, alpha - gamma
        A1 = (wd0 - s2 * (w0 - w_ss)) / (s1 - s2)
        A2 = (w0 - w_ss) - A1
        e1 = np.exp(np.clip(s1 * t, -500, 500))
        e2 = np.exp(np.clip(s2 * t, -500, 500))
        omega[:] = w_ss + A1 * e1 + A2 * e2
        omega_dot[:] = s1 * A1 * e1 + s2 * A2 * e2
    else:  # critically damped
        A = w0 - w_ss
        Bc = wd0 - alpha * A
        eat = np.exp(np.clip(alpha * t, -500, 500))
        omega[:] = w_ss + (A + Bc * t) * eat
        omega_dot[:] = (Bc + alpha * (A + Bc * t)) * eat

    return omega, omega_dot


def evaluate(R, L, J, B, kt, ke):
    """Return (rmse_accel, rmse_decel) for a parameter set."""
    alpha = -B / (2 * J) - R / (2 * L)
    wn_sq = (R * B + ke * kt) / (J * L)
    beta_sq = wn_sq - alpha ** 2
    w_ss = kt * V / (R * B + ke * kt)

    va, _ = solve_phase(t_acc, alpha, beta_sq, w_ss, 0.0, 0.0)
    w_d, wd_d = solve_phase(np.array([T_SWITCH]), alpha, beta_sq, w_ss, 0.0, 0.0)
    vd, _ = solve_phase(t_dec, alpha, beta_sq, 0.0, w_d[0], wd_d[0])

    if np.any(np.isnan(va)) or np.any(np.isnan(vd)):
        return 1e9, 1e9

    ra = np.sqrt(np.mean((v_acc - va) ** 2))
    rd = np.sqrt(np.mean((v_dec - vd) ** 2))
    return ra, rd


# --- Optimizer: random search + local refinement ---
rng = np.random.default_rng(42)

# Start from the current best guess
x0 = np.log10([2.143, 1.61e-4, 5.93e-4, 3.07e-3, 0.0651, 0.0372])

# Bounds in log10 space
bounds_lo = np.log10([0.01, 1e-4, 1e-4, 1e-4, 0.001, 0.001])
bounds_hi = np.log10([5.0,  0.5,  0.1,  0.1,  0.1,   0.3])

names = ['R', 'L', 'J', 'B', 'kt', 'ke']


def total_cost(log_params):
    p = 10 ** log_params
    ra, rd = evaluate(*p)
    # Balance: minimize worst-case RMSE, with w_ss near 91
    w_ss = p[4] * V / (p[0] * p[3] + p[5] * p[4])
    penalty = max(0, abs(w_ss - 91) - 3)
    return max(ra, rd) + 0.3 * min(ra, rd) + penalty


best_cost = total_cost(x0)
best_x = x0.copy()
ra, rd = evaluate(*(10 ** x0))
print(f"Starting point: RMSE accel={ra:.2f}, decel={rd:.2f}, total={ra + rd:.2f}")
print(f"  R={10**x0[0]:.4f}, L={10**x0[1]:.2e}, J={10**x0[2]:.2e}, "
      f"B={10**x0[3]:.2e}, kt={10**x0[4]:.4f}, ke={10**x0[5]:.4f}\n")

# Phase 1: broad random search (500k samples)
print("Phase 1: broad random search...")
N1 = 500_000
samples = rng.uniform(bounds_lo, bounds_hi, (N1, 6))
for i in range(N1):
    c = total_cost(samples[i])
    if c < best_cost:
        best_cost = c
        best_x = samples[i].copy()
        if i % 50000 == 0:
            p = 10 ** best_x
            print(f"  [{i}] cost={best_cost:.3f}  R={p[0]:.3f} L={p[1]:.2e} "
                  f"J={p[2]:.2e} B={p[3]:.2e} kt={p[4]:.4f} ke={p[5]:.4f}")

p = 10 ** best_x
ra, rd = evaluate(*p)
print(f"After phase 1: accel={ra:.2f}, decel={rd:.2f}, total={best_cost:.2f}\n")

# Phase 2: local refinement around best (shrinking perturbations)
print("Phase 2: local refinement...")
for stage in range(10):
    scale = 0.15 * (0.7 ** stage)
    improved = 0
    for _ in range(200_000):
        trial = best_x + rng.normal(0, scale, 6)
        trial = np.clip(trial, bounds_lo, bounds_hi)
        c = total_cost(trial)
        if c < best_cost:
            best_cost = c
            best_x = trial.copy()
            improved += 1
    p = 10 ** best_x
    ra, rd = evaluate(*p)
    print(f"  stage {stage} (scale={scale:.4f}): cost={best_cost:.3f} "
          f"accel={ra:.2f} decel={rd:.2f} improved={improved}")

# --- Final result ---
p = 10 ** best_x
ra, rd = evaluate(*p)

print("\n" + "=" * 60)
print("BEST PARAMETERS FOUND")
print("=" * 60)
print(f"R  = {p[0]:.4f}")
print(f"L  = {p[1]:.6f}")
print(f"J  = {p[2]:.6f}")
print(f"B  = {p[3]:.6f}")
print(f"kt = {p[4]:.5f}")
print(f"ke = {p[5]:.5f}")
print(f"\nRMSE accel = {ra:.2f}")
print(f"RMSE decel = {rd:.2f}")
print(f"RMSE total = {ra + rd:.2f}")

alpha = -p[3] / (2 * p[2]) - p[0] / (2 * p[1])
wn_sq = (p[0] * p[3] + p[5] * p[4]) / (p[2] * p[1])
w_ss = p[4] * 12 / (p[0] * p[3] + p[5] * p[4])
print(f"\nw_ss = {w_ss:.2f}")
print(f"alpha = {alpha:.3f}")
print(f"wn = {np.sqrt(wn_sq):.3f}")

print(f"\n# Paste into main.py:")
print(f"R0, L0, J0, B0 = {p[0]:.4f}, {p[1]:.6f}, {p[2]:.6f}, {p[3]:.6f}")
print(f"kt0, ke0 = {p[4]:.5f}, {p[5]:.5f}")
