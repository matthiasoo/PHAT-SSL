import numpy as np
from pathlib import Path
from scipy.io import wavfile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.animation as animation

geom = Path('geom/rect_16.xml')

def gcc_phat(sig1, sig2):
    SIG1 = np.fft.rfft(sig1)
    SIG2 = np.fft.rfft(sig2)

    R = SIG1 * np.conj(SIG2)
    R_phat = R / (np.abs(R) + 1e-15)

    cc = np.fft.irfft(R_phat)
    cc = np.fft.fftshift(cc)

    return cc

def extract_pos(num):
    root = ET.parse(geom).getroot()
    point = root.find(f"./pos[@Name='Point {num}']")

    pos = np.array([
        float(point.get("x")),
        float(point.get("y"))
    ])
    return pos

def make_pairs(pairs):

    pairs_data = []

    for p in pairs:
        tmp_pos_A = extract_pos(p[0]+1)
        tmp_pos_B = extract_pos(p[1]+1)

        middle = (tmp_pos_A + tmp_pos_B) / 2
        distance = np.linalg.norm(tmp_pos_A - tmp_pos_B)

        pairs_data.append((tmp_pos_A, tmp_pos_B, middle, distance))

    return pairs_data

c = 343.0

signal = Path('signal/rect_16_circle.wav')

fs, data = wavfile.read(signal)

pairs = [(8, 14), (3, 12), (0, 9)]
pairs_data = make_pairs(pairs)

bs = 2048
overlap = int(0.5 * bs)
num_blocks = (data.shape[0] - bs) // overlap
window = np.hanning(bs)

# Plot
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(f"GCC-PHAT", fontsize=14)
fig.subplots_adjust(wspace=0.35)

# Traj
ax0.set_title("Location (X, Y) at Z = 10m")
ax0.set_xlim(-10, 10)
ax0.set_ylim(-10, 10)
ax0.set_aspect('equal')
ax0.grid(True, linestyle='--', alpha=0.6)
ax0.set_xlabel("X [m]")
ax0.set_ylabel("Y [m]")

# Sound source
ss_dot, = ax0.plot([], [], 'ro', markersize=10, label="Sound Source")
ss_path, = ax0.plot([], [], 'r-', alpha=0.3, lw=2)

# Angle
ax1.set_title("Direction of Arrival (DOA)")
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xlabel("X [m]")
ax1.set_ylabel("Y [m]")
ax1.tick_params(axis='x', rotation=45)

# Mics
for i, (p, pd) in enumerate(zip(pairs, pairs_data)):
    label = "Microphones" if i == 0 else None

    ax0.scatter([pd[0][0]], [pd[0][1]], c='black', s=20, label=label)
    ax0.scatter([pd[1][0]], [pd[1][1]], c='black', s=20)

    ax1.scatter([pd[0][0]], [pd[0][1]], c='black', s=50, label=label)
    ax1.scatter([pd[1][0]], [pd[1][1]], c='black', s=50)

# GCC-PHAT
ax2.set_title("Time Difference of Arrival (TDOA)")
ax2.set_xlim(-100, 100)
ax2.set_ylim(-0.2, 1.2)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_xlabel("Delay [sample number]")
ax2.set_ylabel("Amplitude")

# Delay
delay_axis = np.arange(-bs // 2, bs // 2)
line_corr, = ax2.plot([], [], 'k-', alpha=0.7)
peak_dot, = ax2.plot([], [], 'ro', markersize=8)

colors = ['green', 'orange', 'purple']
num_pairs = 3

lines_angle = []
lines_corr = []
peak_dots = []
history_x = []
history_y = []

for i in range(num_pairs):
    l_angle, = ax1.plot([], [], '-', color=colors[i], lw=3, label=f"Pair {pairs[i]}")
    lines_angle.append(l_angle)

    l_corr, = ax2.plot([], [], '-', color=colors[i], alpha=0.5, label=f"Pair {pairs[i]}")
    lines_corr.append(l_corr)

    p_dot, = ax2.plot([], [], 'o', color=colors[i], markersize=6)
    peak_dots.append(p_dot)

ax0.legend(loc="upper right", fontsize=8)
ax1.legend(loc="upper right", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

def update(block):
    start = block * overlap
    end = start + bs

    calculations = []

    D_matrix = []
    V_vector = []

    for i, pd in enumerate(pairs_data):
        data_A = data[start:end, pairs[i][0]] * window
        data_B = data[start:end, pairs[i][1]] * window

        corr = gcc_phat(data_A, data_B)
        peak_index = np.argmax(corr)

        shift_in_samples = peak_index - (bs // 2)
        tau = shift_in_samples / fs

        val = np.clip((c * tau) / pd[3], -1.0, 1.0)
        angle = np.degrees(np.arccos(val))

        vec_AB = pd[1] - pd[0]
        D_matrix.append([vec_AB[0], vec_AB[1]])
        V_vector.append(-c * tau)

        lines_corr[i].set_data(delay_axis, corr)
        peak_dots[i].set_data([shift_in_samples], [corr[peak_index]])
        calculations.extend([lines_corr[i], peak_dots[i]])

        plot_angle = np.arctan2(vec_AB[1], vec_AB[0]) + np.radians(angle)

        L = 3.0

        dx = L * np.cos(plot_angle)
        dy = L * np.sin(plot_angle)

        lines_angle[i].set_data([pd[2][0] - dx, pd[2][0] + dx], [pd[2][1] - dy, pd[2][1] + dy])
        calculations.append(lines_angle[i])

    D_matrix = np.array(D_matrix)
    V_vector = np.array(V_vector)

    U_xy, _, _, _ = np.linalg.lstsq(D_matrix, V_vector, rcond=None)
    ux, uy = U_xy[0], U_xy[1]

    u_norm_sq = ux ** 2 + uy ** 2
    if u_norm_sq > 0.99:
        ux = ux / np.sqrt(u_norm_sq) * 0.99
        uy = uy / np.sqrt(u_norm_sq) * 0.99
        u_norm_sq = ux ** 2 + uy ** 2

    uz = np.sqrt(1.0 - u_norm_sq)

    H = 10.0
    X_drone = H * (ux / uz)
    Y_drone = H * (uy / uz)

    history_x.append(X_drone)
    history_y.append(Y_drone)

    if len(history_x) > 50:
        history_x.pop(0)
        history_y.pop(0)

    ss_dot.set_data([X_drone], [Y_drone])
    ss_path.set_data(history_x, history_y)

    calculations.extend([ss_dot, ss_path])

    return calculations

FPS = fs / overlap
ani = animation.FuncAnimation(fig, update, frames=num_blocks, interval=1000/FPS, blit=True)
plt.show()