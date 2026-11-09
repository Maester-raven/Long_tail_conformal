import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_data

data = load_data()

counts = data["train_counts"]
sorted_counts = np.sort(counts)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    np.arange(1, len(sorted_counts) + 1),
    sorted_counts,
    linewidth=2,
)

ax.set_yscale("log")
ax.set_xlabel("Class rank (sorted by training frequency)")
ax.set_ylabel("Number of training samples (log scale)")
ax.set_title("Pl@ntNet-Trunc: Long-Tailed Class Distribution")
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("class_distribution.png", dpi=200)
plt.close(fig)

print("Saved: class_distribution.png")
print("Classes:", len(counts))
print("Training samples:", counts.sum())
print("Min samples per class:", counts.min())
print("Max samples per class:", counts.max())
print("Classes with zero calibration samples:",
      np.sum(data["cal_counts"] == 0))
