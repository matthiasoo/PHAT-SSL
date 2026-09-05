import numpy as np
from pathlib import Path
from scipy.io import wavfile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def gcc_phat(sig1, sig2):
    SIG1 = np.fft.rfft(sig1)
    SIG2 = np.fft.rfft(sig2)

    R = SIG1 * np.conj(SIG2)
    R_phat = R / (np.abs(R) + 1e-15)

    cc = np.fft.irfft(R_phat)
    cc = np.fft.fftshift(cc)

    return cc

def extract_pos(geom, num):
    root = ET.parse(geom).getroot()
    point = root.find(f"./pos[@Name='Point {num}']")

    pos = np.array([
        float(point.get("x")),
        float(point.get("y"))
    ])
    return pos

def pair(ch1, ch2):
    distance = np.linalg.norm(ch1 - ch2)
    return distance

c = 343.0

geom = Path('geom/rect_16.xml')
signal = Path('signal/rect_16_circle.wav')

fs, data = wavfile.read(signal)

ch_A = 0
ch_B = 15

pos_A = extract_pos(geom, ch_A+1)
pos_B = extract_pos(geom, ch_B+1)
d_AB = pair(pos_A, pos_B)

bs = 2048
overlap = int(0.5 * bs)
num_blocks = (data.shape[0] - bs) // overlap
window = np.hanning(bs)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"GCC-PHAT: Pair {ch_A} & {ch_B}", fontsize=14)

# Map
ax1.set_title("Trajectory / Angle")
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.6)

# Mics
ax1.scatter([pos_A[0]], [pos_A[1]], c='red', s=100, label=f"Mic A ({ch_A})")
ax1.scatter([pos_B[0]], [pos_B[1]], c='blue', s=100, label=f"Mic B ({ch_B})")
ax1.legend(loc="upper right")

# DOA
line_angle, = ax1.plot([], [], 'g-', lw=3, label="DOA")
middle = (pos_A + pos_B) / 2

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

def update(block):
    start = block * overlap
    end = start + bs

    data_A = data[start:end, ch_A] * window
    data_B = data[start:end, ch_B] * window

    corr = gcc_phat(data_A, data_B)
    peak_index = np.argmax(corr)

    shift_in_samples = peak_index - (bs // 2)
    tau = shift_in_samples / fs

    val = np.clip((c * tau) / d_AB, -1.0, 1.0)
    angle = np.degrees(np.arccos(val))

    line_corr.set_data(delay_axis, corr)
    peak_dot.set_data([shift_in_samples], [corr[peak_index]])

    L = 0.08

    vec_AB = pos_B - pos_A
    base_angle = np.arctan2(vec_AB[1], vec_AB[0])
    plot_angle = base_angle + np.radians(angle)

    dx = L * np.cos(plot_angle)
    dy = L * np.sin(plot_angle)
    line_angle.set_data([middle[0], middle[0] + dx], [middle[1], middle[1] + dy])

    return line_corr, peak_dot, line_angle

FPS = fs / overlap
ani = animation.FuncAnimation(fig, update, frames=num_blocks, interval=1000/FPS, blit=True)
plt.show()