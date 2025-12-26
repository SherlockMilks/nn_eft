from os.path import split

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

df = pd.read_csv("output.csv")

methods = ["Hand-calculated", "Ascend", "Descend"]
original_str = df["Original"].to_numpy()
original = []

for row in original_str:
    split = row.split(",")
    original.append([float(j) for j in split])
original = np.array(original)

results = {}
stored = {}

for method in methods:
    values_str = df[method].to_numpy()
    values = []

    for row in values_str:
        split = row.split(",")
        values.append([float(j) for j in split])
    values = np.array(values)

    diff = values - original
    abs_diff = np.abs(diff)
    rel_diff = np.where(original != 0, abs_diff / np.abs(original) * 100, 0)

    stored[method] = values
    results[method] = {
        "Átlagos eltérés": np.mean(abs_diff),
        "Max eltérés": np.max(abs_diff),
        "Szórás": np.std(abs_diff),
        "Relatív eltérés (%)": np.mean(rel_diff),
        "Argmax egyezés (%)": np.mean(np.argmax([original, values], axis=0) == 0) * 100
    }

diff_hand_ascend = stored["Hand-calculated"] - stored["Ascend"]
diff_hand_descend = stored["Hand-calculated"] - stored["Descend"]
diff_ascend_descend = stored["Ascend"] - stored["Descend"]
abs_hand_ascend = np.abs(diff_hand_ascend)
abs_hand_descend = np.abs(diff_hand_descend)
abs_ascend_descend = np.abs(diff_ascend_descend)
rel_hand_ascend = np.where(stored["Hand-calculated"] != 0, abs_hand_ascend / np.abs(stored["Hand-calculated"]) * 100, 0)
rel_hand_descend = np.where(stored["Hand-calculated"] != 0, abs_hand_descend / np.abs(stored["Hand-calculated"]) * 100, 0)
rel_ascend_descend = np.where(stored["Ascend"] != 0, abs_ascend_descend / np.abs(stored["Ascend"]) * 100, 0)


print("------ Tensorflow eredméytől való eltérések: ------")
for method, stats in results.items():
    print(f"\n--- {method} ---")
    for key, value in stats.items():
        print(f"{key}: {value}")

print("\n\n------ Manuálisan számolt, alap sorrendű eredméytől való eltérések: ------")
print("\n--- Ascend ---")
print("Átlagos eltérés:", np.mean(abs_hand_ascend))
print("Max eltérés:", np.max(abs_hand_ascend))
print("Szórás:", np.std(abs_hand_ascend))
print("Relatív eltérés (%):", np.mean(rel_hand_ascend))
print("Argmax egyezés (%):", np.mean(np.argmax([stored["Hand-calculated"], stored["Ascend"]], axis=0) == 0) * 100)

print("\n--- Descend ---")
print("Átlagos eltérés:", np.mean(abs_hand_descend))
print("Max eltérés:", np.max(abs_hand_descend))
print("Szórás:", np.std(abs_hand_descend))
print("Relatív eltérés (%):", np.mean(rel_hand_descend))
print("Argmax egyezés (%):", np.mean(np.argmax([stored["Hand-calculated"], stored["Descend"]], axis=0) == 0) * 100)

print("\n\n------ Növekvő és csökkenő közötti különbségek: ------\n")
print("Átlagos eltérés:", np.mean(abs_ascend_descend))
print("Max eltérés:", np.max(abs_ascend_descend))
print("Szórás:", np.std(abs_ascend_descend))
print("Relatív eltérés (%):", np.mean(rel_ascend_descend))
print("Argmax egyezés (%):", np.mean(np.argmax([stored["Ascend"], stored["Descend"]], axis=0) == 0) * 100)

#Original vs a többi
plt.figure(figsize=(10, 6))
for method in methods:
    diff = original - stored[method]
    plt.hist(diff, bins=100)
    plt.xlabel(f"{method} eltérése az Original-hoz képest")
    plt.ylabel("Gyakoriság")
    plt.title("Eltérések eloszlása")
    plt.show()

#Manual vs ascend
plt.figure(figsize=(10, 6))
diff = stored["Hand-calculated"] - stored["Ascend"]
plt.hist(diff, bins=100)
plt.xlabel("Ascend eltérése a Hand-calculatedhez képest")
plt.ylabel("Gyakoriság")
plt.title("Eltérések eloszlása")
plt.show()

#Manual vs descend
plt.figure(figsize=(10, 6))
diff = stored["Hand-calculated"] - stored["Descend"]
plt.hist(diff, bins=100)
plt.xlabel("Descend eltérése a Hand-calculatedhez képest")
plt.ylabel("Gyakoriság")
plt.title("Eltérések eloszlása")
plt.show()

#Ascend vs descend
plt.figure(figsize=(10, 6))
diff = stored["Ascend"] - stored["Descend"]
plt.hist(diff, bins=100)
plt.xlabel("Descend eltérése az Ascendhez képest")
plt.ylabel("Gyakoriság")
plt.title("Eltérések eloszlása")
plt.show()
