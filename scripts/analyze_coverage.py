import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from src.data.data_loader import load_data
from src.data.split import get_split
from src.evaluation.metrics import evaluate

# 1. Load data and use the existing experimental protocol
data = load_data()
split = get_split(data, budget=1.0, seed=0)

cal_probs = split["calib_probs"]
cal_labels = split["calib_labels"]
test_probs = split["test_probs"]
test_labels = split["test_labels"]
train_counts = data["train_counts"]

# 2. Compute the global conformal threshold
alpha = 0.1
n = len(cal_labels)

# Only use each calibration sample's TRUE class score
scores = 1.0 - cal_probs[np.arange(n), cal_labels]

k = int(np.ceil((n + 1) * (1 - alpha)))

if k > n:
    threshold = np.inf
else:
    threshold = np.sort(scores)[k - 1]

# 3. Construct prediction sets on independent test data
prediction_sets = (1.0 - test_probs) <= threshold

# 4. Evaluate using Task B's existing metrics
result = evaluate(
    prediction_sets,
    test_labels,
    train_counts
)

# 5. Define common and rare classes using training frequency
n_classes = data["n_classes"]
order = np.argsort(train_counts, kind="stable")

rare_classes = order[:n_classes // 3]
common_classes = order[-(n_classes // 3):]

# Equal weight for every class within each group
rare_coverage = result["class_coverage"][rare_classes].mean()
common_coverage = result["class_coverage"][common_classes].mean()

rare_size = result["class_set_size"][rare_classes].mean()
common_size = result["class_set_size"][common_classes].mean()

print("=== Global Conformal Coverage Analysis ===")
print(f"Budget: 100%, Seed: 0")
print(f"Calibration samples: {n}")
print(f"Target coverage: {1-alpha:.1%}")
print(f"Global threshold: {threshold:.6f}")
print()
print(f"Overall weighted coverage: {result['overall_coverage']:.4%}")
print(f"Macro coverage: {result['macro_coverage']:.4%}")
print(f"Common-class coverage: {common_coverage:.4%}")
print(f"Rare-class coverage: {rare_coverage:.4%}")
print(f"Coverage gap: {common_coverage - rare_coverage:.4%}")
print(f"Average prediction set size: {result['avg_set_size']:.4f}")
print(f"Common-class mean set size: {common_size:.4f}")
print(f"Rare-class mean set size: {rare_size:.4f}")

# 6. Generate a comparison figure
fig, ax = plt.subplots(figsize=(7, 5))

values = [common_coverage * 100, rare_coverage * 100]
bars = ax.bar(
    ["Common classes", "Rare classes"],
    values,
    color=["#2774A6", "#D39A4A"]
)

ax.axhline(
    90, color="gray", linestyle="--",
    label="Target coverage (90%)"
)

ax.set_ylim(0, 105)
ax.set_ylabel("Macro coverage within group (%)")
ax.set_title("Global Conformal: Common vs Rare Classes")
ax.legend()

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

fig.tight_layout()

output = Path("results/figures/common_vs_rare_coverage.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=200)
plt.close(fig)

print(f"\nSaved: {output}")
