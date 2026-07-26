import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import Utils
from matplotlib.font_manager import FontProperties
from pathlib import Path
import os

np.set_printoptions(precision=64)
DTYPE = np.float64

def process_csv(input_csv, adversarial=False):
    print(2*"\n" + 70 * "-")
    print(input_csv)
    print(70 * "-")

    data = []
    data_eft = []

    # The indexes of the largest logits
    logit1_idx = 0
    logit2_idx = 0

    logit1_greater = 0
    logit2_greater = 0
    logits_equal = 0

    logit1_greater_eft = 0
    logit2_greater_eft = 0
    logits_equal_eft = 0

    with open(input_csv, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

        i = 0
        while i < len(lines):
            label = lines[i]

            if label == "Ascend":
                raw = lines[i + 1].split(":")[1]
                eft = lines[i + 2].split(":")[1]

                ascend_output = list(map(float, raw.split(",")))
                ascend_output_eft = list(map(float, eft.split(",")))

                (_, logit1_idx), (_, logit2_idx) = Utils.two_largest(ascend_output)

                i += 3

            elif label == "Descend":
                raw = lines[i + 1].split(":")[1]
                eft = lines[i + 2].split(":")[1]

                descend_output = list(map(float, raw.split(",")))
                descend_output_eft = list(map(float, eft.split(",")))

                i += 3

            elif label.startswith("Random"):
                raw = lines[i + 1].split(":")[1]
                eft = lines[i + 2].split(":")[1]

                nums = list(map(float, raw.split(",")))
                nums_eft = list(map(float, eft.split(",")))

                data.append(nums)
                data_eft.append(nums_eft)

                if nums[logit1_idx] > nums[logit2_idx]:
                    logit1_greater += 1
                elif nums[logit1_idx] < nums[logit2_idx]:
                    logit2_greater += 1
                else:
                    logits_equal += 1

                if nums_eft[logit1_idx] > nums_eft[logit2_idx]:
                    logit1_greater_eft += 1
                elif nums_eft[logit1_idx] < nums_eft[logit2_idx]:
                    logit2_greater_eft += 1
                else:
                    logits_equal_eft += 1

                i += 3

            else:
                i += 1

    file_name_split = input_csv.split("/")
    if adversarial:
        output_dir = Path(f"output/single_img_result/{file_name_split[1]}/{file_name_split[2]}/{file_name_split[3]}/{file_name_split[4]}")
        file_name = file_name_split[5].replace('.csv', '.txt')
    else:
        output_dir = Path(f"output/single_img_result/{file_name_split[1]}/{file_name_split[2]}/{file_name_split[3]}")
        file_name = file_name_split[4].replace('.csv', '.txt')

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / file_name

    datasets = [data, data_eft]
    raw_data = []

    with open(output_file, "w", encoding="utf-8") as g:
        log = Utils.log_factory(g)
        for i, current_data in enumerate(datasets):
            current_data = np.array(current_data, dtype=DTYPE)

            all_cols_min, all_cols_max, all_cols_diff, all_cols_median = [], [], [], []

            for j in range(0, 10):
                col = current_data[:, j]

                all_cols_max.append(np.max(col))
                all_cols_min.append(np.min(col))
                all_cols_diff.append(np.ptp(col))
                all_cols_median.append(np.median(col))

            if i == 0:
                log("\nRAW")
                log(50 * "-")
                diff = np.array(ascend_output) - np.array(descend_output)
                log(f"Ascend, descend diff for each output: {diff.tolist()}")
            else:
                log("\nEFT")
                log(50 * "-")
                diff = np.array(ascend_output_eft) - np.array(descend_output_eft)
                log(f"Ascend, descend diff for each output: {diff.tolist()}")


            log("Smallest value for each output:", all_cols_min)
            log("Largest value for each output:", all_cols_max)
            log("Range for each output:", all_cols_diff)
            log("Median for each output:", all_cols_median)
            log(50 * "-")

            if i == 0:
                raw_data = current_data
                log(f"First logit is larger: {logit1_greater} ({logit1_idx})")
                log(f"Second logit is larger: {logit2_greater} ({logit2_idx})")
                log(f"The two logits are equal: {logits_equal}")
            else:
                log(f"First logit is larger: {logit1_greater_eft} ({logit1_idx})")
                log(f"Second logit is larger: {logit2_greater_eft} ({logit2_idx})")
                log(f"The two logits are equal: {logits_equal_eft}")

            log(70 * "-")

    return raw_data, ascend_output, descend_output


input_dir = "runs/adversarial/bfloat16/sequential/base_first/"
input_files = os.listdir(input_dir)

for file in input_files:

    input_sequential = input_dir + file
    input_pairwise = input_sequential.replace("sequential", "pairwise")

    is_adversarial = True
    sequential_output, ascend_output, descend_output = process_csv(input_sequential, is_adversarial)
    pairwise_output, _, _ = process_csv(input_pairwise, is_adversarial)


    for i in range(0,10):
        #Frequency
        combined = np.concatenate((sequential_output[:, i], pairwise_output[:, i]))
        counts, bin_edges = np.histogram(combined, bins=100)
        mode = 0.5 * (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1])

        shifted_pairwise = pairwise_output[0:, i] - mode
        shifted_sequential = sequential_output[0:, i] - mode
        shifted_ascend = ascend_output[i] - mode
        shifted_descend = descend_output[i] - mode

        common_min = min(shifted_sequential.min(), shifted_pairwise.min())
        common_max = max(shifted_sequential.max(), shifted_pairwise.max())

        bins = np.linspace(common_min, common_max, 51)

        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.figure(figsize=(8, 5))

        plt.hist(shifted_sequential, bins=bins, alpha=0.5, label="Sequential", color="purple")
        plt.hist(shifted_pairwise, bins=bins, alpha=0.5, color="orange", label="Pairwise")

        plt.axvline(float(shifted_ascend), color="red", linewidth=1.5, alpha=0.65, label="Ascend")
        plt.axvline(float(shifted_descend), color="green", linewidth=1.5, alpha=0.65, label="Descend")

        plt.xlabel("Deviation from mode", fontsize=13)
        plt.ylabel("Frequency", fontsize=13)
        plt.legend(prop=FontProperties(size='11'))

        input_sequential_parts = input_sequential.split('/')
        if is_adversarial:
            output_dir = Path(f"output/pair_vs_seq/{input_sequential_parts[1]}/{input_sequential_parts[2]}/{input_sequential_parts[4]}/{Path(input_sequential_parts[5]).stem}")
        else:
            output_dir = Path(f"output/pair_vs_seq/{input_sequential_parts[1]}/{input_sequential_parts[2]}/{Path(input_sequential_parts[4]).stem}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"plot{i}.pdf"

        plt.savefig(output_file)
        plt.close()

