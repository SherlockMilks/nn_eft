"""
Script used to make good-looking histograms
for sequential vs pairwise comparison
"""

import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties

matplotlib.use("PDF")
import matplotlib.pyplot as plt


pairwise = np.load("par.npy")
sequential = np.load("seq.npy")
orig = np.load("original.npy")
ascend = np.load("ascend.npy")
descend = np.load("descend.npy")

combined = np.concatenate([sequential, pairwise])
counts, bin_edges = np.histogram(combined, bins=100)
mode = 0.5 * (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1])

shifted_pairwise = pairwise - mode
shifted_sequential = sequential - mode
shifted_orig = orig - mode
shifted_ascend = ascend - mode
shifted_descend = descend - mode

common_min = min(shifted_sequential.min(), shifted_pairwise.min())
common_max = max(shifted_sequential.max(), shifted_pairwise.max())

bins = np.linspace(common_min, common_max, 51)

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.figure(figsize=(8, 4))

plt.hist(shifted_sequential, bins=bins, alpha=0.5, label="Sequential", color="purple")
plt.hist(shifted_pairwise, bins=bins, alpha=0.5, color="orange", label="Pairwise")

plt.axvline(float(shifted_orig), color="blue", linewidth=1.5, label="Original")
plt.axvline(float(shifted_ascend), color="red", linewidth=1.5, label="Ascend")
plt.axvline(float(shifted_descend), color="green", linewidth=1.5, label="Descend")

plt.xlabel("Deviation from mode", fontweight='bold', fontsize=12)
plt.ylabel("Frequency", fontweight='bold', fontsize=12)
# plt.title("Float64 model deviations for output 0", fontweight='bold', fontsize=12)
plt.legend(prop=FontProperties(weight='bold'))

plt.savefig("plot0.pdf")
