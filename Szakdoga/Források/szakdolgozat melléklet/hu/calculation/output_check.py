import random
from collections import Counter
def load_errors(file_path):
    errors = {}
    with open(file_path, "r") as f:
        next(f)  # fejléc átugrása
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                index, pred, actual = map(int, parts)
                errors[index] = (pred, actual)
    return errors
def get_top_misclassifications(errors, top_n=5):
    misclassifications = [f"{actual} → {pred}" for pred, actual in errors.values()]
    counter = Counter(misclassifications)
    return counter.most_common(top_n)

# Betöltés
errors1 = load_errors("target/original_errors_detailed.txt")
# errors2 = load_errors("second_errors_detailed.txt")
errors2 = load_errors("target/gpu_errors_detailed.txt")
# errors2 = load_errors("cpu_errors_detailed.txt")

# Alap statisztikák
total_errors1 = len(errors1)
total_errors2 = len(errors2)

top_errors1 = get_top_misclassifications(errors1)
top_errors2 = get_top_misclassifications(errors2)

# Közös hibák
common_indexes = set(errors1.keys()) & set(errors2.keys())

# Elemzés
same_misclassifications = []
diff_misclassifications = []

for idx in common_indexes:
    pred1, actual1 = errors1[idx]
    pred2, actual2 = errors2[idx]
    if pred1 == pred2:
        same_misclassifications.append((idx, pred1, actual1))
    else:
        diff_misclassifications.append((idx, pred1, pred2, actual1))

random.seed(42)
for i in range(10):
    print(f"{i}. random: {random.randint(1, 100)}")
# Kiírás konzolra
print(f"Hibák száma Model1-ben: {total_errors1}")
print(f"Hibák száma Model2-ben: {total_errors2}")
print(f"Közös hibás indexek száma: {len(common_indexes)}")
print(f"Ugyanarra hibáztak: {len(same_misclassifications)}")
print(f"Különbözőre hibáztak: {len(diff_misclassifications)}")

print("\nModel1 – Top 5 félreosztályozás:")
for cls, count in top_errors1:
    print(f"{cls}: {count} alkalom")

print("\nModel2 – Top 5 félreosztályozás:")
for cls, count in top_errors2:
    print(f"{cls}: {count} alkalom")