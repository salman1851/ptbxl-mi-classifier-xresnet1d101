import os
import csv
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, auc,
                             precision_recall_curve)
from dataset import ECGDataset
from params import *
from xresnet1d import xresnet1d50, xresnet1d101
from matplotlib.colors import PowerNorm

MODEL_FACTORY = {"xresnet1d50": xresnet1d50, "xresnet1d101": xresnet1d101}

def _div(a, b):
    """Safe division: nan instead of a crash when a class has an empty denominator."""
    return float(a / b) if b > 0 else float("nan")


def _clean(o):
    """Recursively replace nan with None so metrics.json is strictly valid JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def _predict(model, loader, device):
    """Run the model over a loader; return softmax probabilities and true labels."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(yb.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def _compute_metrics(probs, labels, class_names):
    """Confusion matrix, one-vs-rest metrics per class, macro averages, overall scores."""
    K = len(class_names)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(K)))
    total = cm.sum()

    per_class = {}
    for k, name in enumerate(class_names):
        # one-vs-rest counts for class k, read off the confusion matrix
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp

        sens = _div(tp, tp + fn)   # recall
        spec = _div(tn, tn + fp)
        ppv = _div(tp, tp + fp)    # precision
        npv = _div(tn, tn + fn)

        # threshold-free metrics use the probabilities, not the argmax predictions
        y_bin = (labels == k).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, k])
        per_class[name] = {
            "support": int(tp + fn),
            "prevalence": float((tp + fn) / total),
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "f1": _div(2 * ppv * sens, ppv + sens),
            "auc": float(roc_auc_score(y_bin, probs[:, k])),
            "auprc": float(auc(rec, prec)),   # trapezoidal area under the PR curve
        }

    # macro = unweighted mean over classes, so "normal" can't dominate
    metric_names = ["sensitivity", "specificity", "ppv", "npv", "f1", "auc", "auprc"]
    macro = {m: float(np.nanmean([per_class[n][m] for n in class_names])) for m in metric_names}
    overall = {
        "accuracy": float(np.trace(cm) / total),
        "balanced_accuracy": macro["sensitivity"],   # mean per-class recall
        "macro_f1": macro["f1"],
    }
    return cm, per_class, macro, overall


def _plot_confusion(cm, class_names, path):
    """Two panels: counts on a plain grid (support reference, no color-mapped
    heatmap -- counts span too wide a range under class imbalance to color
    meaningfully), and row-normalized (recall), which is the heatmap that
    actually carries color-coded information."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    n = len(class_names)

    # --- left panel: counts, plain grid, no heatmap ---
    ax = axes[0]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)   # row 0 at top, matching imshow's orientation
    ax.set_aspect("equal")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="black")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Counts")

    # --- right panel: row-normalized recall, the actual heatmap ---
    ax = axes[1]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Row-normalized (recall)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(cm_norm[i, j], ".2f"), ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_pr(probs, labels, per_class, class_names, path):
    """One-vs-rest ROC and PR curves, with the argmax operating point marked on each."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for k, name in enumerate(class_names):
        c = f"C{k}"
        m = per_class[name]
        y_bin = (labels == k).astype(int)

        fpr, tpr, _ = roc_curve(y_bin, probs[:, k])
        ax_roc.plot(fpr, tpr, color=c, label=f"{name} (AUC {m['auc']:.3f})")
        ax_roc.plot(1 - m["specificity"], m["sensitivity"], "o", color=c, markeredgecolor="black")

        prec, rec, _ = precision_recall_curve(y_bin, probs[:, k])
        ax_pr.plot(rec, prec, color=c, label=f"{name} (AUPRC {m['auprc']:.3f})")
        ax_pr.plot(m["sensitivity"], m["ppv"], "o", color=c, markeredgecolor="black")
        ax_pr.axhline(m["prevalence"], color=c, ls=":", lw=1)   # no-skill baseline = prevalence

    # Legend-only entries; the order added here is the order shown in the legend
    ax_roc.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Chance")
    ax_roc.plot([], [], "o", color="gray", markeredgecolor="black", label="Operating point (argmax)")
    ax_pr.plot([], [], color="gray", ls=":", label="No-skill baseline (prevalence)")
    ax_pr.plot([], [], "o", color="gray", markeredgecolor="black", label="Operating point (argmax)")

    ax_roc.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
               title="ROC (one-vs-rest)", xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)",
              title="Precision-Recall (one-vs-rest)", xlim=(0, 1), ylim=(0, 1.02))
    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is ax_roc else "lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_csv(path, class_names, per_class, macro, overall):
    """Write the console table to CSV: one row per class plus a macro row.
    The three overall scores are filled in on the macro row only."""
    metric_cols = ["sensitivity", "specificity", "ppv", "npv", "f1", "auc", "auprc"]
    header = ["class", "n"] + metric_cols + ["accuracy", "balanced_accuracy", "macro_f1"]

    def fmt(v):   # nan -> empty cell
        return "" if v is None or np.isnan(v) else f"{v:.4f}"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name in class_names:
            m = per_class[name]
            w.writerow([name, m["support"]] + [fmt(m[c]) for c in metric_cols] + ["", "", ""])
        n_total = sum(per_class[n]["support"] for n in class_names)
        w.writerow(["macro", n_total] + [fmt(macro[c]) for c in metric_cols]
                   + [fmt(overall["accuracy"]), fmt(overall["balanced_accuracy"]), fmt(overall["macro_f1"])])


def evaluate(out_dir, split="test", batch_size=BATCH_SIZE):
    """
    Evaluate out_dir/best_model_checkpoint.pt on one split of PROCESSED_DATA_DIR.

    out_dir: the experiment folder written by train()
    split:   "test" (final report) or "val" (model selection / sanity checks)

    Writes into out_dir/eval_<split>/:
      probs.npy, labels.npy, metrics.json, metrics_table.csv,
      confusion_matrices.png, roc_pr.png
    and returns the metrics dict.
    """
    eval_dir = os.path.join(out_dir, f"eval_{split}")
    os.makedirs(eval_dir, exist_ok=True)
    class_names = sorted(CLASS_MAP, key=CLASS_MAP.get)   # names ordered by integer label

    # --- load best checkpoint (dropout has no weights, so fc_drop doesn't matter here) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(out_dir, "best_model_checkpoint.pt")
    with open(os.path.join(out_dir, "history.json")) as f:
        model_depth = json.load(f)["config"]["model_depth"]
    model = MODEL_FACTORY[model_depth](in_ch=IN_CH, c_out=NUM_CLASSES, stem_k=STEM_K, block_k=BLOCK_K)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    # --- predict (no shuffle, so probs stay aligned with labels) ---
    loader = DataLoader(ECGDataset(split), batch_size=batch_size)
    probs, labels = _predict(model, loader, device)
    assert len(np.unique(labels)) == NUM_CLASSES, "a class is missing from this split"
    np.save(os.path.join(eval_dir, "probs.npy"), probs)     # keep raw predictions so any
    np.save(os.path.join(eval_dir, "labels.npy"), labels)   # metric can be recomputed later

    # --- metrics + figures ---
    cm, per_class, macro, overall = _compute_metrics(probs, labels, class_names)
    _plot_confusion(cm, class_names, os.path.join(eval_dir, "confusion_matrices.png"))
    _plot_roc_pr(probs, labels, per_class, class_names, os.path.join(eval_dir, "roc_pr.png"))

    metrics = {
        "split": split,
        "checkpoint": ckpt_path,
        "n_samples": int(len(labels)),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro": macro,
        "overall": overall,
    }
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(_clean(metrics), f, indent=2)

    _save_csv(os.path.join(eval_dir, "metrics_table.csv"), class_names, per_class, macro, overall)

    # --- console summary ---
    print(f"\nEvaluation on '{split}' set ({len(labels)} samples)")
    print(f"{'class':<16}{'n':>6}{'sens':>8}{'spec':>8}{'PPV':>8}{'NPV':>8}{'F1':>8}{'AUC':>8}{'AUPRC':>8}")
    for name in class_names:
        m = per_class[name]
        print(f"{name:<16}{m['support']:>6}{m['sensitivity']:>8.3f}{m['specificity']:>8.3f}"
              f"{m['ppv']:>8.3f}{m['npv']:>8.3f}{m['f1']:>8.3f}{m['auc']:>8.3f}{m['auprc']:>8.3f}")
    print(f"{'macro':<16}{'':>6}{macro['sensitivity']:>8.3f}{macro['specificity']:>8.3f}"
          f"{macro['ppv']:>8.3f}{macro['npv']:>8.3f}{macro['f1']:>8.3f}{macro['auc']:>8.3f}{macro['auprc']:>8.3f}")
    print(f"accuracy = {overall['accuracy']:.4f}, balanced accuracy = {overall['balanced_accuracy']:.4f}, "
          f"macro-F1 = {overall['macro_f1']:.4f}")

    # free memory so repeated runs in one process don't accumulate
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


if __name__ == "__main__":
    evaluate("results")