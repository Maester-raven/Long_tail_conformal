import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.data_loader import load_data

data = load_data()

probs = data["test_probs"]
labels = data["test_labels"]
train_counts = data["train_counts"]
cal_counts = data["cal_counts"]
n_classes = data["n_classes"]

# Each prediction is the most probable class.
predictions = np.argmax(probs, axis=1)
correct = predictions == labels

# Accuracy for each true class.
test_counts = np.bincount(labels, minlength=n_classes)
correct_counts = np.bincount(
    labels,
    weights=correct.astype(float),
    minlength=n_classes
)
class_accuracy = correct_counts / test_counts

# Rare classes: bottom third by training frequency.
rare_classes = np.argsort(train_counts, kind="stable")[:n_classes // 3]
zero_cal_classes = np.where(cal_counts == 0)[0]

print("Top-1 Accuracy:", f"{correct.mean():.4%}")
print("Macro Accuracy:", f"{class_accuracy.mean():.4%}")
print("Rare-class Accuracy:", f"{class_accuracy[rare_classes].mean():.4%}")
print("Zero-calibration Accuracy:",
      f"{class_accuracy[zero_cal_classes].mean():.4%}")
print("Zero-calibration classes:", len(zero_cal_classes))

# Analyze relationship between frequency and accuracy.
order = np.argsort(train_counts)
sorted_counts = train_counts[order]
sorted_accuracy = class_accuracy[order]

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(
    sorted_counts,
    sorted_accuracy,
    alpha=0.6,
    s=18
)
ax.set_xscale("log")
ax.set_xlabel("Training samples per class (log scale)")
ax.set_ylabel("Top-1 accuracy per class")
ax.set_title("Class Frequency vs. Model Accuracy")
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("results/figures/frequency_vs_accuracy.png", dpi=200)
plt.close(fig)

print("Saved: frequency_vs_accuracy.png")
