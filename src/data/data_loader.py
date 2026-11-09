from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "plantnet-trunc"
PREFIX = "best-plantnet-trunc-model"


def load_data():
    data = {
        "cal_probs": np.load(DATA_DIR / f"{PREFIX}_cal_softmax.npy"),
        "cal_labels": np.load(DATA_DIR / f"{PREFIX}_cal_labels.npy"),
        "test_probs": np.load(DATA_DIR / f"{PREFIX}_test_softmax.npy"),
        "test_labels": np.load(DATA_DIR / f"{PREFIX}_test_labels.npy"),
        "train_labels": np.load(DATA_DIR / "plantnet-trunc_train_labels.npy"),
    }

    n_classes = data["cal_probs"].shape[1]

    assert data["test_probs"].shape[1] == n_classes
    assert len(data["cal_probs"]) == len(data["cal_labels"])
    assert len(data["test_probs"]) == len(data["test_labels"])

    for name in ("cal", "test"):
        probs = data[f"{name}_probs"]
        labels = data[f"{name}_labels"]

        assert np.isfinite(probs).all()
        assert np.all((probs >= 0) & (probs <= 1))
        assert np.allclose(probs.sum(axis=1), 1)
        assert np.all((labels >= 0) & (labels < n_classes))

    assert np.all((data["train_labels"] >= 0) &
                  (data["train_labels"] < n_classes))

    data["train_counts"] = np.bincount(
        data["train_labels"], minlength=n_classes
    )
    data["cal_counts"] = np.bincount(
        data["cal_labels"], minlength=n_classes
    )
    data["n_classes"] = n_classes

    return data


if __name__ == "__main__":
    data = load_data()

    print("Number of classes:", data["n_classes"])
    print("Calibration samples:", len(data["cal_labels"]))
    print("Test samples:", len(data["test_labels"]))
    print("Training samples:", len(data["train_labels"]))
    print("Classes without calibration:", np.sum(data["cal_counts"] == 0))
    print("Training count min/max:",
          data["train_counts"].min(),
          data["train_counts"].max())
    print("Test count min/max:",
          np.bincount(
              data["test_labels"],
              minlength=data["n_classes"]
          ).min(),
          np.bincount(
              data["test_labels"],
              minlength=data["n_classes"]
          ).max())
