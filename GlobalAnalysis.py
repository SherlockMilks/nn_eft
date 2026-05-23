import numpy as np
import matplotlib
matplotlib.use("TKagg")
import matplotlib.pyplot as plt
import os
import csv

np.set_printoptions(precision=64)
DTYPE = np.float64

dir = "output/eft/norm2/sequential/logit"
raw_all = []
eft_all = []

files_sorted = sorted(os.listdir(dir), key=lambda x: int(x.split("logit")[1].split(".")[0]))
for file in files_sorted:
    path = os.path.join(dir, file)

    raw_img = []
    eft_img = []

    with open(path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith("Raw:"):
                raw_img.append(list(map(float, line[4:].split(','))))
            if line.startswith("EFT:"):
                eft_img.append(list(map(float, line[4:].split(','))))

    raw_all.append(np.array(raw_img))
    eft_all.append(np.array(eft_img))


raw_all = np.array(raw_all, dtype=DTYPE)
eft_all = np.array(eft_all, dtype=DTYPE)



raw_img_ranges = []
eft_img_ranges = []
raw_zerorange_count = 0
eft_zerorange_count = 0

for raw_img, eft_img in zip(raw_all, eft_all):
    raw_ranges = np.max(raw_img, axis=0) - np.min(raw_img, axis=0)
    eft_ranges = np.max(eft_img, axis=0) - np.min(eft_img, axis=0)

    raw_max_range = np.max(raw_ranges)
    eft_max_range = np.max(eft_ranges)
    if raw_max_range == 0.0:
        raw_zerorange_count += 1
    if eft_max_range == 0.0:
        eft_zerorange_count += 1

    raw_img_ranges.append(raw_max_range)
    eft_img_ranges.append(eft_max_range)


raw_img_ranges = np.array(raw_img_ranges)
eft_img_ranges = np.array(eft_img_ranges)


with open("output/csv/norm2_scatter.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "raw_range", "eft_range"])

    for i in range(len(raw_img_ranges)):
        writer.writerow([i, raw_img_ranges[i], eft_img_ranges[i]])

print("=== RAW ===")
raw_min_idx = np.argmin(raw_img_ranges)
raw_max_idx = np.argmax(raw_img_ranges)

print("min range:", raw_img_ranges[raw_min_idx], "at image:", raw_min_idx)
print("max range:", raw_img_ranges[raw_max_idx], "at image:", raw_max_idx)
print("mean range:", np.mean(raw_img_ranges))
print("deterministic outputs across permutations:", raw_zerorange_count)

print("\n=== EFT ===")
eft_min_idx = np.argmin(eft_img_ranges)
eft_max_idx = np.argmax(eft_img_ranges)

print("min range:", eft_img_ranges[eft_min_idx], "at image:", eft_min_idx)
print("max range:", eft_img_ranges[eft_max_idx], "at image:", eft_max_idx)
print("mean range:", np.mean(eft_img_ranges))
print("deterministic outputs across permutations:", eft_zerorange_count)