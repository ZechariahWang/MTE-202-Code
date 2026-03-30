import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import csv
import math

# --- Load measured data from CSV ---
time_ms = []
velocity = []

with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        time_ms.append(int(row['time_ms']))
        velocity.append(int(row['velocity']))

time_s = [t / 1000.0 for t in time_ms]

# Split data at t=1.0s into acceleration and deceleration phases
t_switch = 1.0
accel_t = [t for t in time_s if t <= t_switch]
accel_v = [velocity[i] for i, t in enumerate(time_s) if t <= t_switch]

decel_t = [t - t_switch for t in time_s if t > t_switch]
decel_v = [velocity[i] for i, t in enumerate(time_s) if t > t_switch]


# --- First-order DC motor model ---
# tau * dw/dt + w = k * u
# w(t) = k*u + (w0 - k*u) * exp(-t/tau)
#
# Acceleration:  w0=0, u=input  =>  w(t) = k*u * (1 - exp(-t/tau))
# Deceleration:  w0=91, u=0     =>  w(t) = w0 * exp(-t/tau)

def model_accel(t_arr, k, tau, u):
    w0 = 0.0
    return [k * u + (w0 - k * u) * math.exp(-t / tau) for t in t_arr]


def model_decel(t_arr, k, tau, w0):
    # u=0, so k*u=0 => w(t) = w0 * exp(-t/tau)
    u = 0.0
    return [k * u + (w0 - k * u) * math.exp(-t / tau) for t in t_arr]


def rmse(measured, modeled):
    n = len(measured)
    return math.sqrt(sum((m - p) ** 2 for m, p in zip(measured, modeled)) / n)


# --- Initial parameter values ---
k0 = 0.909
tau0 = 0.047
u = 100.0
w0_decel = 91.0

vel_accel = model_accel(accel_t, k0, tau0, u)
vel_decel = model_decel(decel_t, k0, tau0, w0_decel)

# --- Set up figure ---
fig, (ax_acc, ax_dec) = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(bottom=0.22, hspace=0.4)

color_data = '#888888'
color_accel = '#2563eb'
color_decel = '#dc2626'

# Acceleration subplot
ax_acc.scatter(accel_t, accel_v, s=10, color=color_data, label='Measured', zorder=5)
line_acc, = ax_acc.plot(accel_t, vel_accel, color=color_accel, linewidth=2, label='Model')
ax_acc.set_ylabel('Velocity', fontsize=12)
ax_acc.set_title('Acceleration Phase (w0=0)', fontsize=12)
ax_acc.legend(loc='lower right')
ax_acc.grid(True, alpha=0.3)
rmse_acc_text = ax_acc.text(0.98, 0.05, '', transform=ax_acc.transAxes,
                            ha='right', fontsize=11, color=color_accel)

# Deceleration subplot
ax_dec.scatter(decel_t, decel_v, s=10, color=color_data, label='Measured', zorder=5)
line_dec, = ax_dec.plot(decel_t, vel_decel, color=color_decel, linewidth=2, label='Model')
ax_dec.set_xlabel('Time (s) from switch', fontsize=12)
ax_dec.set_ylabel('Velocity', fontsize=12)
ax_dec.set_title(f'Deceleration Phase (w0={w0_decel})', fontsize=12)
ax_dec.legend(loc='upper right')
ax_dec.grid(True, alpha=0.3)
rmse_dec_text = ax_dec.text(0.98, 0.05, '', transform=ax_dec.transAxes,
                            ha='right', fontsize=11, color=color_decel)

fig.suptitle('First-Order DC Motor Model Tuning', fontsize=14, fontweight='bold')

# --- Sliders ---
ax_k = plt.axes([0.15, 0.10, 0.70, 0.03])
ax_tau = plt.axes([0.15, 0.05, 0.70, 0.03])

slider_k = Slider(ax_k, 'k', 0.0, 2.0, valinit=k0, valstep=0.001)
slider_tau = Slider(ax_tau, 'τ (tau)', 0.01, 1.0, valinit=tau0, valstep=0.001)


def update(_):
    k = slider_k.val
    tau = slider_tau.val

    va = model_accel(accel_t, k, tau, u)
    vd = model_decel(decel_t, k, tau, w0_decel)

    line_acc.set_ydata(va)
    line_dec.set_ydata(vd)

    all_acc = list(accel_v) + va
    all_dec = list(decel_v) + vd
    ax_acc.set_ylim(min(all_acc) - 5, max(all_acc) + 5)
    ax_dec.set_ylim(min(all_dec) - 5, max(all_dec) + 5)

    rmse_acc_text.set_text(f'RMSE: {rmse(accel_v, va):.2f}')
    rmse_dec_text.set_text(f'RMSE: {rmse(decel_v, vd):.2f}')

    fig.canvas.draw_idle()


slider_k.on_changed(update)
slider_tau.on_changed(update)

update(None)
plt.show()
