import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from params import *


def preprocess(seed=RANDOM_SEED):
    """
    Load the raw signals, cap the "normal" class at NORMAL_SAMPLE_SIZE, make a
    stratified train/val/test split, and write the six .npy files into
    PROCESSED_DATA_DIR, overwriting any existing ones.

    seed: random seed for both the normal-class subsample and the split
          (a fresh seed per experiment = a different subsample and split)

    Returns the per-class sample counts of each split, e.g.
    {"train": [n_class0, n_class1, n_class2], "val": [...], "test": [...]}.
    """
    rng = np.random.default_rng(seed)

    # --- collect file paths and integer labels, capping the "normal" class ---
    files, labels = [], []
    for cls_name, cls_id in CLASS_MAP.items():
        cls_files = glob.glob(os.path.join(DATA_PATH, cls_name, "*.npy"))
        if cls_name == "normal" and len(cls_files) > NORMAL_SAMPLE_SIZE:
            cls_files = rng.choice(cls_files, size=NORMAL_SAMPLE_SIZE, replace=False).tolist()
        files += cls_files
        labels += [cls_id] * len(cls_files)
    assert len(files) > 0, f"no .npy files found under {DATA_PATH}"
    y = np.array(labels)

    # --- load into one preallocated float32 array (no list -> array copy) ---
    X = np.empty((len(files), IN_CH, SEQ_LEN), dtype=np.float32)
    for i, f in enumerate(files):
        sig = np.load(f).astype(np.float32).T   # raw files are (5000, 12); transpose to (12, 5000)
        assert sig.shape == (IN_CH, SEQ_LEN), f"unexpected shape {sig.shape} in {f}"
        X[i] = sig

    # --- stratified split on indices (train / val / test) ---
    idx = np.arange(len(y))
    idx_temp, idx_test = train_test_split(
        idx, test_size=TRAIN_VAL_TEST[2], stratify=y, random_state=seed)
    val_ratio = TRAIN_VAL_TEST[1] / (TRAIN_VAL_TEST[0] + TRAIN_VAL_TEST[1])  # val's share of the remainder
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_ratio, stratify=y[idx_temp], random_state=seed)

    # --- save one split at a time; np.save overwrites existing files ---
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    counts = {}
    for name, split_idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        np.save(os.path.join(PROCESSED_DATA_DIR, f"X_{name}.npy"), X[split_idx])
        np.save(os.path.join(PROCESSED_DATA_DIR, f"y_{name}.npy"), y[split_idx])
        counts[name] = np.bincount(y[split_idx], minlength=NUM_CLASSES).tolist()

    del X   # release the big array before training allocates
    return counts


if __name__ == "__main__":
    print(preprocess())