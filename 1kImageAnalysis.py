import numpy as np
import os
import csv
from pathlib import Path
import Utils

np.set_printoptions(precision=64)
DTYPE = np.float64

dir = "runs/eft/f64/sequential/logit"
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



raw_img_rmax = []
raw_img_rmedian = []
raw_zerorange_count = 0

eft_img_rmax = []
eft_img_rmedian = []
eft_zerorange_count = 0

for raw_img, eft_img in zip(raw_all, eft_all):

    raw_ranges = np.ptp(raw_img, axis=0)
    eft_ranges = np.ptp(eft_img, axis=0)

    raw_rmax = np.max(raw_ranges)
    raw_rmedian = np.median(raw_ranges)

    eft_rmax = np.max(eft_ranges)
    eft_rmedian = np.median(eft_ranges)

    if raw_rmax == 0.0:
        raw_zerorange_count += 1
    if eft_rmax == 0.0:
        eft_zerorange_count += 1

    raw_img_rmax.append(raw_rmax)
    raw_img_rmedian.append(raw_rmedian)

    eft_img_rmax.append(eft_rmax)
    eft_img_rmedian.append(eft_rmedian)


raw_img_rmax = np.array(raw_img_rmax)
raw_img_rmedian = np.array(raw_img_rmedian)

eft_img_rmax = np.array(eft_img_rmax)
eft_img_rmedian = np.array(eft_img_rmedian)


# with open("output/scatter/norm2_scatter.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["image_id", "raw_rmax", "eft_rmax"])
#
#     for i in range(len(raw_img_ranges)):
#         writer.writerow([i, raw_img_rmax[i], eft_img_rmax[i]])


output_dir = Path("output/1k_img_results")
output_dir.mkdir(parents=True, exist_ok=True)

dir_name = dir.split("/")
file_name = "_".join(dir_name[1:4])
output_file = output_dir / f"{Path(file_name).stem}.txt"

with open(output_file, "a", encoding="utf-8") as f:
    log = Utils.log_factory(f)

    log("=== RAW ===")
    raw_rmax_min_idx = np.argmin(raw_img_rmax)
    raw_rmax_max_idx = np.argmax(raw_img_rmax)

    log("RMAX min:", raw_img_rmax[raw_rmax_min_idx], "at image:", raw_rmax_min_idx)
    log("RMAX max:", raw_img_rmax[raw_rmax_max_idx], "at image:", raw_rmax_max_idx)
    log("RMAX median:", np.median(raw_img_rmax))
    log("RMAX mean:", np.mean(raw_img_rmax))

    raw_rmedian_max_idx = np.argmax(raw_img_rmedian)

    log("RMEDIAN max:", raw_img_rmedian[raw_rmedian_max_idx], "at image:", raw_rmedian_max_idx)
    log("RMEDIAN median:", np.median(raw_img_rmedian))
    log("RMEDIAN mean:", np.mean(raw_img_rmedian))

    log("deterministic outputs across permutations:", raw_zerorange_count)


    log("\n=== EFT ===")
    eft_rmax_min_idx = np.argmin(eft_img_rmax)
    eft_rmax_max_idx = np.argmax(eft_img_rmax)

    log("RMAX min:", eft_img_rmax[eft_rmax_min_idx], "at image:", eft_rmax_min_idx)
    log("RMAX max:", eft_img_rmax[eft_rmax_max_idx], "at image:", eft_rmax_max_idx)
    log("RMAX median:", np.median(eft_img_rmax))
    log("RMAX mean:", np.mean(eft_img_rmax))

    eft_rmedian_max_idx = np.argmax(eft_img_rmedian)

    log("RMEDIAN max:", eft_img_rmedian[eft_rmedian_max_idx], "at image:", eft_rmedian_max_idx)
    log("RMEDIAN median:", np.median(eft_img_rmedian))
    log("RMEDIAN mean:", np.mean(eft_img_rmedian))

    log("deterministic outputs across permutations:", eft_zerorange_count)