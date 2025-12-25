import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=64)

input = "output/first_output_modelf64.csv"

data = []
labels = []
ascend = []
descend = []

with open(input, 'r') as f:
    lines = f.readlines()

    for i, line in enumerate(lines):
        if i in (2, 3, 4, 5):
            print(line)
            if i == 3:
                ascend = list(map(float, line.split(',')))
            if i == 5:
                descend = list(map(float, line.split(',')))
            continue
        line = line.strip().strip('"')
        if i % 2 == 0:
            labels.append(line)
        else:
            nums = list(map(float, line.split(',')))
            data.append(nums)

data = np.array(data, dtype=np.float64)

col_max = np.max(data, axis=0)
col_min = np.min(data, axis=0)
col_diff = col_max - col_min

print("\nLegkisebb értékek kimenetenként:", col_min)
print("Legnagyobb értékek kimenetenként:", col_max)
print("Terjedelem kimenetenként:", col_diff)
print("------------------------------------------------------------------------------")


for i in range(10):
    output_value = data[0:, i]

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
    shifted_orig = float(data[0, i]) - center
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


# diff = data - data[0]
#
# #Összes eltérés egyben
# plt.figure(figsize=(10, 5))
# plt.hist(diff[1:].flatten(), bins=30, density=True, edgecolor='black', alpha=0.6)
# sns.kdeplot(diff[1:].flatten(), fill=True, edgecolor='black', alpha=0.5)
# plt.title("Eltérések az eredeti sorrendhez képest")
# plt.xlabel("Eltérés nagysága")
# plt.ylabel("Sűrűség")
# plt.show()
