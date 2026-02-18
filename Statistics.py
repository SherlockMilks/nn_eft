import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import Utils

np.set_printoptions(precision=64)

input = "output/basic/f64/modelf64_sequential_worstimg_logit.csv"

data = []
data_eft = []
labels = []
tf = []
original = []
ascend = []
descend = []

logit1_idx = 0
logit2_idx = 0

logit1_win = 0
logit2_win = 0
tie = 0

logit1_win_eft = 0
logit2_win_eft = 0
tie_eft = 0

with open(input, 'r') as f:
    lines = f.readlines()
    has_eft = False

    it_lines = iter(range(len(lines)))
    for i in it_lines:

        if (not has_eft and i % 2 == 0) or (has_eft and i % 3 == 0):
            line_strip = lines[i].strip()
            if line_strip == "Original":
                original = list(map(float, lines[i+1].split(',')))

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
                    logit1_win_eft += 1
                elif nums_eft[logit1_idx] < nums_eft[logit2_idx]:
                    logit2_win_eft += 1
                else:
                    tie_eft += 1

                next(it_lines, None)

            else:
                line = lines[i].strip().strip('"')
                nums = list(map(float, line.split(',')))
                data.append(nums)

            if nums[logit1_idx] > nums[logit2_idx]:
                logit1_win += 1
            elif nums[logit1_idx] < nums[logit2_idx]:
                logit2_win += 1
            else:
                tie += 1


datasets = [data]
if logit1_win_eft+logit2_win_eft+tie_eft != 0:
    datasets.append(data_eft)

for i, current_data in enumerate(datasets):
    current_data = np.array(current_data, dtype=np.float64)

    col_max = np.max(current_data, axis=0)
    col_min = np.min(current_data, axis=0)
    col_diff = col_max - col_min

    print("\nLegkisebb értékek kimenetenként:", col_min)
    print("Legnagyobb értékek kimenetenként:", col_max)
    print("Terjedelem kimenetenként:", col_diff)
    print("------------------------------------------------------------------------------")

    if i == 0:
        print(f"Első logit nagyobb: {logit1_win} ({logit1_idx})" )
        print(f"Második logit nagyobb: {logit2_win} ({logit2_idx})")
        print(f"Két logit egyenlő: {tie}")
        print("------------------------------------------------------------------------------")
    else:
        print(f"Első logit nagyobb: {logit1_win_eft} ({logit1_idx})")
        print(f"Második logit nagyobb: {logit2_win_eft} ({logit2_idx})")
        print(f"Két logit egyenlő: {tie_eft}")
        print("------------------------------------------------------------------------------")


    for i in range(10):
        output_value = current_data[0:, i]

        max_output = np.max(output_value)
        max_pos = np.argmax(output_value)
        max_method = labels[0:][max_pos]

        min_output = np.min(output_value)
        min_pos = np.argmin(output_value)
        min_method = labels[0:][min_pos]

        print(f"\nKimenet {i}:")
        print(f"Legnagyobb érték : {max_output}")
        print(f"Sorrend: {max_method}")

        print(f"Legkisebb érték : {min_output}")
        print(f"Sorrend: {min_method}")

        #Gyakoriságok
        counts, bin_edges = np.histogram(output_value, bins=100)
        center = 0.5 * (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1])
        print("center:",center)
        shifted = output_value - center
        shifted_orig = float(original[i]) - center
        shifted_ascend = float(ascend[i]) - center
        shifted_descend = float(descend[i]) - center

        plt.figure(figsize=(10, 5))
        plt.hist(shifted, bins=50, density=True, edgecolor='black', alpha=0.6, range=(min_output-center, max_output-center))
        sns.kdeplot(shifted, fill=True, edgecolor='black', alpha=0.5)
        plt.axvline(float(shifted_orig), color="blue", linewidth=1.5, alpha=0.5, label="Eredeti sorrend")
        plt.axvline(float(shifted_ascend), color="red", linewidth=1.5, alpha=0.5, label="Növekvő sorrend")
        plt.axvline(float(shifted_descend), color="green", linewidth=1.5, alpha=0.5, label="Csökkenő sorrend")
        plt.title(f"Kimenet {i} érékeinek eloszlása")
        plt.xlabel("Érték eltérése a középértéktől")
        plt.ylabel("Sűrűség")
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.show()

