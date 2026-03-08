import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import Utils

np.set_printoptions(precision=64)

input = "output/adversarial/f64/modelf64_sequential_09img_logit.csv"

data = []
data_eft = []
labels = []
tf = []
original = []
ascend = []
descend = []

# The indexes of the largest logits
logit1_idx = 0
logit2_idx = 0

logit1_greater = 0
logit2_greater = 0
logits_equal = 0

logit1_greater_eft = 0
logit2_greater_eft = 0
logits_equal_eft = 0

with open(input, 'r') as f:
    lines = f.readlines()
    has_eft = False

    it_lines = iter(range(len(lines)))
    for i in it_lines:

        if (not has_eft and i % 2 == 0) or (has_eft and i % 3 == 0):
            line_strip = lines[i].strip()
            if line_strip == "Original":
                original = list(map(float, lines[i+1].split(',')))

                # All files start with the original order so this is a great place to find the largest logits
                (_, logit1_idx), (_, logit2_idx) = Utils.two_largest(original)

                next(it_lines, None)

            elif line_strip == "Ascend":
                ascend = list(map(float, lines[i+1].split(',')))
                next(it_lines, None)

            elif line_strip == "Descend":
                descend = list(map(float, lines[i+1].split(',')))
                next(it_lines, None)

            elif line_strip == "Tensorflow":
                tf = list(map(float, lines[i+1].split(',')))
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
if logit1_greater_eft+logit2_greater_eft+logits_equal_eft != 0:
    datasets.append(data_eft)

for i, current_data in enumerate(datasets):
    current_data = np.array(current_data, dtype=np.float64)

    col_max = np.max(current_data, axis=0)
    col_min = np.min(current_data, axis=0)
    col_diff = col_max - col_min

    print("\nSmallest value for each output:", col_min)
    print("Largest value for each output:", col_max)
    print("Range for each output:", col_diff)

    print("------------------------------------------------------------------------------")

    if i == 0:
        print(f"First logit is larger: {logit1_greater} ({logit1_idx})")
        print(f"Second logit is larger: {logit2_greater} ({logit2_idx})")
        print(f"The two logits are equal: {logits_equal}")
        print("------------------------------------------------------------------------------")
    else:
        print(f"First logit is larger: {logit1_greater_eft} ({logit1_idx})")
        print(f"Second logit is larger: {logit2_greater_eft} ({logit2_idx})")
        print(f"The two logits are equal: {logits_equal_eft}")
        print("------------------------------------------------------------------------------")


    for i in range(10):
        output_value = current_data[0:, i]

        max_output = col_max[i]
        max_pos = np.argmax(max_output)
        max_method = labels[0:][max_pos]

        min_output = col_min[i]
        min_pos = np.argmin(output_value)
        min_method = labels[0:][min_pos]

        print(f"\nOutput {i}:")
        print(f"Largest value : {np.array([max_output])}")
        print(f"Order label: {max_method}")

        print(f"Smallest value : {np.array([min_output])}")
        print(f"Order label: {min_method}")

        #Frequency
        counts, bin_edges = np.histogram(output_value, bins=100)
        mean = 0.5 * (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1])
        shifted = output_value - mean
        shifted_orig = original[i] - mean
        shifted_ascend = ascend[i] - mean
        shifted_descend = descend[i] - mean

        plt.figure(figsize=(10, 5))
        plt.hist(shifted, bins=50, edgecolor='black', alpha=0.6, range=(min_output-mean, max_output-mean))
        plt.axvline(float(shifted_orig), color="blue", linewidth=1.5, alpha=0.5, label="Original")
        plt.axvline(float(shifted_ascend), color="red", linewidth=1.5, alpha=0.5, label="Ascend")
        plt.axvline(float(shifted_descend), color="green", linewidth=1.5, alpha=0.5, label="Descend")
        plt.title(f"Histogram of deviations for output {i}")
        plt.xlabel("Deviation from the mean")
        plt.ylabel("Frequency")
        plt.show()

