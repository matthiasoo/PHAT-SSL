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
        vec_AB = tmp_pos_B - tmp_pos_A
        base_angle = np.arctan2(vec_AB[1], vec_AB[0])

        pairs_data.append((tmp_pos_A, tmp_pos_B, middle, distance, base_angle))

    return pairs_data

c = 343.0

signal = Path('signal/rect_16_circle.wav')

fs, data = wavfile.read(signal)

pairs = [(0, 15), (3, 12), (0, 9)]
pairs_data = make_pairs(pairs)

bs = 2048
overlap = int(0.5 * bs)
num_blocks = (data.shape[0] - bs) // overlap
window = np.hanning(bs)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"GCC-PHAT", fontsize=14)

# Map
ax1.set_title("Trajectory / Angle")
ax1.set_xlim(-1.0, 1.0)
ax1.set_ylim(-1.0, 1.0)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.6)

# Mics
for i, (p, pd) in enumerate(zip(pairs, pairs_data)):
    label = "Microphones" if i == 0 else None

    ax1.scatter([pd[0][0]], [pd[0][1]], c='black', s=50, label=label)
    ax1.scatter([pd[1][0]], [pd[1][1]], c='black', s=50)

# GCC-PHAT
ax2.set_title("Cross-Correlation")
ax2.set_xlabel("Delay (sample)")
ax2.set_ylabel("Amplitude (norm)")
ax2.set_xlim(-100, 100)
ax2.set_ylim(-0.2, 1.2)
ax2.grid(True, linestyle='--', alpha=0.6)

# Delay
delay_axis = np.arange(-bs // 2, bs // 2)
line_corr, = ax2.plot([], [], 'k-', alpha=0.7)
peak_dot, = ax2.plot([], [], 'ro', markersize=8)
text_info = ax2.text(0.05, 0.85, '', transform=ax2.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

colors = ['green', 'orange', 'purple']
num_pairs = 3

lines_angle = []
lines_corr = []
peak_dots = []

for i in range(num_pairs):
    l_angle, = ax1.plot([], [], '-', color=colors[i], lw=3, label=f"Pair {pairs[i]}")
    lines_angle.append(l_angle)

    l_corr, = ax2.plot([], [], '-', color=colors[i], alpha=0.5)
    lines_corr.append(l_corr)

    p_dot, = ax2.plot([], [], 'o', color=colors[i], markersize=6)
    peak_dots.append(p_dot)
ax1.legend(loc="upper right", fontsize=8)

def update(block):
    start = block * overlap
    end = start + bs

    calculations = []

    for i, pd in enumerate(pairs_data):
        data_A = data[start:end, pairs[i][0]] * window
        data_B = data[start:end, pairs[i][1]] * window

        corr = gcc_phat(data_A, data_B)
        peak_index = np.argmax(corr)

        shift_in_samples = peak_index - (bs // 2)
        tau = shift_in_samples / fs

        val = np.clip((c * tau) / pd[3], -1.0, 1.0)
        angle = np.degrees(np.arccos(val))

        lines_corr[i].set_data(delay_axis, corr)
        peak_dots[i].set_data([shift_in_samples], [corr[peak_index]])
        calculations.extend([lines_corr[i], peak_dots[i]])

        plot_angle = pd[4] + np.radians(angle)

        L = 3.0

        dx = L * np.cos(plot_angle)
        dy = L * np.sin(plot_angle)

        lines_angle[i].set_data([pd[2][0] - dx, pd[2][0] + dx], [pd[2][1] - dy, pd[2][1] + dy])
        calculations.append(lines_angle[i])

    return calculations

FPS = fs / overlap
ani = animation.FuncAnimation(fig, update, frames=num_blocks, interval=1000/FPS, blit=True)
plt.show()