import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import Utils
from matplotlib.font_manager import FontProperties
from pathlib import Path

np.set_printoptions(precision=64)

input_sequential = "runs/first_worst/f64/modelf64_sequential_firstimg_logit.csv"
input_pairwise = input_sequential.replace("sequential", "parallel")

DTYPE = np.float64

def process_csv(input_csv):
    print(2*"\n" + 70 * "-")
    print(input_csv)
    print(70 * "-")

    data = []
    data_eft = []
    labels = []

    # The indexes of the largest logits
    logit1_idx = 0
    logit2_idx = 0

    logit1_greater = 0
    logit2_greater = 0
    logits_equal = 0

    logit1_greater_eft = 0
    logit2_greater_eft = 0
    logits_equal_eft = 0

    global original_output, ascend_output, descend_output, tf_output

    with open(input_csv, 'r') as f:
        lines = f.readlines()
        has_eft = False

        it_lines = iter(range(len(lines)))
        for i in it_lines:

            if (not has_eft and i % 2 == 0) or (has_eft and i % 3 == 0):
                line_strip = lines[i].strip()
                if line_strip == "Original":
                    original_output = list(map(float, lines[i+1].split(',')))

                    # All files start with the original order so this is a great place to find the largest logits
                    (_, logit1_idx), (_, logit2_idx) = Utils.two_largest(original_output)

                    next(it_lines, None)

                elif line_strip == "Ascend":
                    ascend_output = list(map(float, lines[i+1].split(',')))
                    next(it_lines, None)

                elif line_strip == "Descend":
                    descend_output = list(map(float, lines[i+1].split(',')))
                    next(it_lines, None)

                elif line_strip == "Tensorflow":
                    tf_output = list(map(float, lines[i+1].split(',')))
                    next(it_lines, None)

                else:
                    labels.append(line_strip)

            else:
                has_eft = lines[i].strip().split(":")[0] == "Raw"
                if has_eft:
                    line = lines[i].strip().split(":")[1].strip('"')
                    line_eft = lines[i+1].strip().split(":")[1].strip('"')

                    nums = list(map(float, line.split(',')))
                    nums_eft = list(map(float, line_eft.split(',')))

                    data.append(nums)
                    data_eft.append(nums_eft)

                    if nums_eft[logit1_idx] > nums_eft[logit2_idx]:
                        logit1_greater_eft += 1
                    elif nums_eft[logit1_idx] < nums_eft[logit2_idx]:
                        logit2_greater_eft += 1
                    else:
                        logits_equal_eft += 1

                    next(it_lines, None)

                else:
                    line = lines[i].strip().strip('"')
                    nums = list(map(float, line.split(',')))
                    data.append(nums)

                if nums[logit1_idx] > nums[logit2_idx]:
                    logit1_greater += 1
                elif nums[logit1_idx] < nums[logit2_idx]:
                    logit2_greater += 1
                else:
                    logits_equal += 1

    datasets = [data]
    if logit1_greater_eft + logit2_greater_eft + logits_equal_eft != 0:
        datasets.append(data_eft)

    raw_data = []
    for i, current_data in enumerate(datasets):
        current_data = np.array(current_data, dtype=DTYPE)

        all_cols_min, all_cols_max, all_cols_diff, all_cols_median = [], [], [], []

        for j in range(0, 10):
            col = current_data[:, j]

            all_cols_max.append(np.max(col))
            all_cols_min.append(np.min(col))
            all_cols_diff.append(np.ptp(col))
            all_cols_median.append(np.median(col))

        output_dir = Path("output/single_img_result")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{Path(input_csv).stem}.txt"

        with open(output_file, "a", encoding="utf-8") as f:
            log = Utils.log_factory(f)

            if i == 0:
                log("\nRAW")
            else:
                log("\nEFT")

            log(50 * "-")
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

    return raw_data

sequential_output = process_csv(input_sequential)
pairwise_output = process_csv(input_pairwise)


for i in range(0,10):
    #Frequency
    combined = np.concatenate((sequential_output[:, i], pairwise_output[:, i]))
    counts, bin_edges = np.histogram(combined, bins=100)
    mode = 0.5 * (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1])

    shifted_pairwise = pairwise_output[0:, i] - mode
    shifted_sequential = sequential_output[0:, i] - mode
    shifted_orig = original_output[i] - mode
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

    plt.axvline(float(shifted_orig), color="blue", linewidth=1.5, alpha=0.65, label="Original")
    plt.axvline(float(shifted_ascend), color="red", linewidth=1.5, alpha=0.65, label="Ascend")
    plt.axvline(float(shifted_descend), color="green", linewidth=1.5, alpha=0.65, label="Descend")

    plt.xlabel("Devation from mode", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.legend(prop=FontProperties(size='11'))

    csv_name = (f"output/pair_vs_seq/"
                f"{input_sequential.split('/')[3].replace('sequential', '').replace('logit.csv', '')}"
                f"plot{i}.pdf")

    plt.savefig(csv_name)

