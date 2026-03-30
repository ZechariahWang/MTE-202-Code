import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import csv

time_ms_list, vel_list = [], []
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        time_ms_list.append(int(row['time_ms']))
        vel_list.append(int(row['velocity']))

t_data = np.array(time_ms_list) / 1000.0
meas_vel = np.array(vel_list, dtype=float)

T_SWITCH = 1.0
W0_DECEL = 91.0  # measured velocity just before decel

accel_mask = t_data <= T_SWITCH
decel_mask = ~accel_mask
t_accel = t_data[accel_mask]
v_accel = meas_vel[accel_mask]
t_decel = t_data[decel_mask] - T_SWITCH
v_decel = meas_vel[decel_mask]

def first_order_accel(t, k, tau, u):
    return k * u * (1 - np.exp(-t / tau))

def first_order_decel(t, tau, w0):
    return w0 * np.exp(-t / tau)

def solve_2nd_phase(t, alpha, beta_sq, w_ss, w0, wd0):
    """Analytical solution for one phase."""
    omega = np.empty_like(t)
    omega_dot = np.empty_like(t)

    if beta_sq > 1e-10:  # underdamped
        beta = np.sqrt(beta_sq)
        A = w0 - w_ss
        C = (wd0 - alpha * A) / beta
        eat = np.exp(alpha * t)
        cb, sb = np.cos(beta * t), np.sin(beta * t)
        omega[:] = w_ss + eat * (A * cb + C * sb)
        omega_dot[:] = eat * ((alpha * A + beta * C) * cb
                              + (alpha * C - beta * A) * sb)
    elif beta_sq < -1e-10:  # overdamped
        gamma = np.sqrt(-beta_sq)
        s1, s2 = alpha + gamma, alpha - gamma
        A1 = (wd0 - s2 * (w0 - w_ss)) / (s1 - s2)
        A2 = (w0 - w_ss) - A1
        e1, e2 = np.exp(s1 * t), np.exp(s2 * t)
        omega[:] = w_ss + A1 * e1 + A2 * e2
        omega_dot[:] = s1 * A1 * e1 + s2 * A2 * e2
    else:  # critically damped
        A = w0 - w_ss
        Bc = wd0 - alpha * A
        eat = np.exp(alpha * t)
        omega[:] = w_ss + (A + Bc * t) * eat
        omega_dot[:] = (Bc + alpha * (A + Bc * t)) * eat

    return omega, omega_dot


def second_order_model(t_acc, t_dec, R, L, J, B, kt, ke, V):
    alpha = -B / (2 * J) - R / (2 * L)
    wn_sq = (R * B + ke * kt) / (J * L)
    beta_sq = wn_sq - alpha ** 2
    wn = np.sqrt(max(wn_sq, 0.0))
    zeta = -alpha / wn if wn > 1e-12 else 999.0

    w_ss = kt * V / (R * B + ke * kt)

    # Accel: ω(0)=0, ω'(0)=0
    va, _ = solve_2nd_phase(t_acc, alpha, beta_sq, w_ss, 0.0, 0.0)

    # State at T_SWITCH
    w_d, wd_d = solve_2nd_phase(np.array([T_SWITCH]), alpha, beta_sq, w_ss, 0.0, 0.0)

    # Decel: V=0, carry over state
    vd, _ = solve_2nd_phase(t_dec, alpha, beta_sq, 0.0, w_d[0], wd_d[0])

    return va, vd, alpha, beta_sq, zeta, wn


# First order
k0, tau0, u0 = 0.9065, 0.1329, 100.0

# Second order
R0, L0, J0, B0 = 2.143, 1.61e-4, 5.93e-4, 3.07e-3
kt0, ke0 = 0.0698, 0.0372

V_SUPPLY = 12.0  

# --- Figure: two subplots (accel top, decel bottom) ---
fig, (ax_acc, ax_dec) = plt.subplots(2, 1, figsize=(14, 9))
plt.subplots_adjust(bottom=0.35, hspace=0.40, top=0.93, left=0.08, right=0.85)

color_data = '#888888'
color_acc = '#2563eb'
color_dec = '#dc2626'

# Scatter measured data (always visible)
ax_acc.scatter(t_accel, v_accel, s=8, color=color_data, label='Measured', zorder=5)
ax_dec.scatter(t_decel, v_decel, s=8, color=color_data, label='Measured', zorder=5)

# Model lines (will be updated)
line_acc, = ax_acc.plot([], [], color=color_acc, lw=2, label='Model')
line_dec, = ax_dec.plot([], [], color=color_dec, lw=2, label='Model')

ax_acc.set_ylabel('Velocity', fontsize=11)
ax_acc.set_title('Acceleration Phase', fontsize=11)
ax_acc.legend(loc='lower right', fontsize=9)
ax_acc.grid(True, alpha=0.3)
rmse_acc_txt = ax_acc.text(0.98, 0.05, '', transform=ax_acc.transAxes,
                           ha='right', fontsize=10, color=color_acc)

ax_dec.set_xlabel('Time (s) from decel start', fontsize=11)
ax_dec.set_ylabel('Velocity', fontsize=11)
ax_dec.set_title('Deceleration Phase', fontsize=11)
ax_dec.legend(loc='upper right', fontsize=9)
ax_dec.grid(True, alpha=0.3)
rmse_dec_txt = ax_dec.text(0.98, 0.05, '', transform=ax_dec.transAxes,
                           ha='right', fontsize=10, color=color_dec)

title_txt = fig.suptitle('First-Order Model', fontsize=13, fontweight='bold')

# Info readout (for 2nd order)
info_txt = ax_dec.text(0.02, 0.95, '', transform=ax_dec.transAxes, va='top',
                       fontsize=9, fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.8))

# --- Radio button to toggle model ---
ax_radio = plt.axes([0.87, 0.75, 0.12, 0.10])
radio = RadioButtons(ax_radio, ('1st Order', '2nd Order'), active=0)

# --- Helper: logarithmic slider ---
def make_log_slider(rect, label, vmin, vmax, vinit, visible=True):
    """Slider that moves linearly in log10 space; .real_val gives actual value."""
    log_min = np.log10(vmin)
    log_max = np.log10(vmax)
    log_init = np.log10(vinit)
    ax_s = plt.axes(rect)
    ax_s.set_visible(visible)
    s = Slider(ax_s, label, log_min, log_max, valinit=log_init,
               valfmt='%.4g')
    # patch the display to show the real value
    s._log_base_min = log_min
    s._log_base_max = log_max
    s.real_val = vinit

    def _fmt(v):
        rv = 10 ** v
        if rv >= 1:
            return f'{rv:.3f}'
        elif rv >= 0.01:
            return f'{rv:.4f}'
        else:
            return f'{rv:.2e}'

    s.valfmt = '%s'
    s.valtext.set_text(_fmt(log_init))

    def _on_changed(val):
        s.real_val = 10 ** val
        s.valtext.set_text(_fmt(val))

    s.on_changed(_on_changed)
    return ax_s, s


# === First-order sliders (log) ===
ax_k, s_k = make_log_slider([0.12, 0.26, 0.33, 0.018], 'k', 0.01, 2.0, k0)
ax_tau, s_tau = make_log_slider([0.12, 0.23, 0.33, 0.018], 'τ (tau)', 0.005, 1.0, tau0)

first_order_sliders = [s_k, s_tau]
first_order_axes = [ax_k, ax_tau]

# === Second-order sliders (log) ===
so_slider_defs = [
    ([0.12, 0.26, 0.33, 0.018], 'R (Ω)',    0.01,  5.0,   R0),
    ([0.12, 0.23, 0.33, 0.018], 'L (H)',     0.0001, 0.5,  L0),
    ([0.12, 0.20, 0.33, 0.018], 'J (kg·m²)', 0.0001, 0.1,  J0),
    ([0.12, 0.17, 0.33, 0.018], 'B (N·m·s)', 0.0001, 0.1,  B0),
    ([0.58, 0.26, 0.33, 0.018], 'k_t',       0.001,  0.1,  kt0),
    ([0.58, 0.23, 0.33, 0.018], 'k_e',       0.001,  0.3,  ke0),
]

second_order_sliders = []
second_order_axes = []
for rect, label, vmin, vmax, vinit in so_slider_defs:
    ax_s, s = make_log_slider(rect, label, vmin, vmax, vinit, visible=False)
    second_order_sliders.append(s)
    second_order_axes.append(ax_s)

s_R, s_L, s_J, s_B, s_kt, s_ke = second_order_sliders

current_mode = '1st Order'


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def refresh():
    if current_mode == '1st Order':
        k, tau = s_k.real_val, s_tau.real_val
        va = first_order_accel(t_accel, k, tau, u0)
        vd = first_order_decel(t_decel, tau, W0_DECEL)

        line_acc.set_data(t_accel, va)
        line_dec.set_data(t_decel, vd)
        info_txt.set_text('')
        title_txt.set_text('First-Order Model')
    else:
        try:
            va, vd, alpha, beta_sq, zeta, wn = second_order_model(
                t_accel, t_decel,
                s_R.real_val, s_L.real_val, s_J.real_val, s_B.real_val,
                s_kt.real_val, s_ke.real_val, V_SUPPLY)
        except (ValueError, ZeroDivisionError, OverflowError):
            return

        va = np.clip(va, -1e6, 1e6)
        vd = np.clip(vd, -1e6, 1e6)

        line_acc.set_data(t_accel, va)
        line_dec.set_data(t_decel, vd)
        if beta_sq >= 0:
            beta_str = f'\u03b2 = {np.sqrt(beta_sq):10.3f}  (cos/sin)'
        else:
            beta_str = f'\u03b3 = {np.sqrt(-beta_sq):10.3f}  (cosh/sinh)'
        info_txt.set_text(
            f'\u03b1 = {alpha:10.3f}\n'
            f'{beta_str}\n'
            f'\u03b6 = {zeta:10.4f}\n'
            f'\u03c9n= {wn:10.3f}')
        title_txt.set_text('Second-Order Model')

    va_arr = line_acc.get_ydata()
    vd_arr = line_dec.get_ydata()
    if len(va_arr) > 0:
        lo = min(np.min(v_accel), np.min(va_arr))
        hi = max(np.max(v_accel), np.max(va_arr))
        ax_acc.set_ylim(lo - 5, hi + 5)
        rmse_acc_txt.set_text(f'RMSE: {rmse(v_accel, np.asarray(va_arr)):.2f}')
    if len(vd_arr) > 0:
        lo = min(np.min(v_decel), np.min(vd_arr))
        hi = max(np.max(v_decel), np.max(vd_arr))
        ax_dec.set_ylim(lo - 5, hi + 5)
        rmse_dec_txt.set_text(f'RMSE: {rmse(v_decel, np.asarray(vd_arr)):.2f}')

    fig.canvas.draw_idle()


def on_slider_change(_):
    refresh()


def on_mode_change(label):
    global current_mode
    current_mode = label

    is_1st = label == '1st Order'
    for ax_s in first_order_axes:
        ax_s.set_visible(is_1st)
    for ax_s in second_order_axes:
        ax_s.set_visible(not is_1st)

    refresh()


radio.on_clicked(on_mode_change)

for s in first_order_sliders:
    s.on_changed(on_slider_change)
for s in second_order_sliders:
    s.on_changed(on_slider_change)

refresh()
plt.show()
