import os
import csv
import json
import numpy as np
from params import *
from evaluate import _compute_metrics, _clean

CLASS_NAMES = sorted(CLASS_MAP, key=CLASS_MAP.get)   # names ordered by integer label


def summarize_runs(results_dir, split="val"):
    """
    Build a one-row-per-experiment summary table from every subfolder of
    results_dir, append mean and SD rows, and identify the run whose
    balanced accuracy is closest to the median (the "representative" run
    for bootstrapping).

    results_dir: a datetime folder from run_experiments.py, containing one
                 subfolder per experiment (each with history.json and
                 eval_<split>/metrics.json)
    split:       which evaluation to read ("val", matching how experiments
                 are compared during model/config selection)

    Writes results_dir/run_summary.csv and returns (rows, median_run_name),
    where rows is the list of per-run dicts (not including the mean/SD rows)
    and median_run_name is the subfolder name of the representative run.
    """
    rows = []
    for name in sorted(os.listdir(results_dir)):
        exp_dir = os.path.join(results_dir, name)
        history_path = os.path.join(exp_dir, "history.json")
        metrics_path = os.path.join(exp_dir, f"eval_{split}", "metrics.json")
        if not (os.path.isfile(history_path) and os.path.isfile(metrics_path)):
            continue   # skip anything that isn't a completed experiment folder (e.g. a failed run)

        with open(history_path) as f:
            history = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)

        macro = metrics["macro"]
        overall = metrics["overall"]
        rows.append({
            "run": name,
            "balanced_accuracy": overall["balanced_accuracy"],
            "macro_f1": overall["macro_f1"],
            "mean_auc": macro["auc"],
            "mean_auprc": macro["auprc"],
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
        })

    assert rows, f"no completed experiments with eval_{split}/metrics.json found under {results_dir}"

    # --- find the run closest to the median balanced accuracy ---
    numeric_cols = ["balanced_accuracy", "macro_f1", "mean_auc", "mean_auprc",
                    "best_epoch", "best_val_loss"]
    bal_accs = np.array([r["balanced_accuracy"] for r in rows])
    median_val = np.median(bal_accs)
    median_idx = np.argmin(np.abs(bal_accs - median_val))
    median_run_name = rows[median_idx]["run"]

    # mark the representative row so it's visible directly in the table, not just printed
    for r in rows:
        r["is_median_run"] = (r["run"] == median_run_name)

    # --- save the table, with mean and SD rows appended at the bottom ---
    fieldnames = ["run"] + numeric_cols + ["is_median_run"]
    out_path = os.path.join(results_dir, "run_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        mean_row = {"run": "mean"}
        sd_row = {"run": "std"}
        for col in numeric_cols:
            vals = np.array([r[col] for r in rows], dtype=float)
            mean_row[col] = float(np.mean(vals))
            sd_row[col] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        mean_row["is_median_run"] = ""
        sd_row["is_median_run"] = ""
        w.writerow(mean_row)
        w.writerow(sd_row)

    # --- console summary ---
    print(f"\n{len(rows)} runs summarized from '{results_dir}' (split='{split}')")
    print(f"{'run':<12}{'bal_acc':>10}{'macro_f1':>10}{'mean_auc':>10}{'mean_auprc':>12}{'best_ep':>9}")
    for r in rows:
        marker = "  <- median" if r["is_median_run"] else ""
        print(f"{r['run']:<12}{r['balanced_accuracy']:>10.4f}{r['macro_f1']:>10.4f}"
              f"{r['mean_auc']:>10.4f}{r['mean_auprc']:>12.4f}{r['best_epoch']:>9}{marker}")
    print(f"Median balanced accuracy: {median_val:.4f}  ->  representative run: {median_run_name}")

    return rows, median_run_name


def _save_ci_csv(path, class_names, per_class, macro, overall):
    """Flat CSV: one row per class plus macro, with 'point [lo, hi]' per metric column."""
    per_class_metrics = ["sensitivity", "specificity", "ppv", "npv", "f1", "auc", "auprc"]
    summary_metrics = ["accuracy", "balanced_accuracy", "macro_f1"]

    def fmt(d):
        if d["lo"] is None:
            return ""
        return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"

    header = ["class"] + per_class_metrics + summary_metrics
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name in class_names:
            row = [name] + [fmt(per_class[name][m]) for m in per_class_metrics] + ["", "", ""]
            w.writerow(row)
        macro_row = ["macro"] + [fmt(macro[m]) for m in per_class_metrics] \
                    + [fmt(overall[m]) for m in summary_metrics]
        w.writerow(macro_row)


def bootstrap_ci(exp_dir, split="test", n_boot=2000, ci=0.95, seed=RANDOM_SEED):
    """
    Bootstrap confidence intervals for one experiment's test predictions.

    exp_dir: the experiment folder (e.g. the median run from summarize_runs)
    split:   which evaluation's predictions to bootstrap ("test", for the
             final report)
    n_boot:  number of bootstrap resamples
    ci:      confidence level (0.95 -> 2.5th/97.5th percentiles)
    seed:    for reproducible resampling

    Writes exp_dir/eval_<split>/bootstrap_ci.json and returns the dict of
    {metric_path: {"point": ..., "lo": ..., "hi": ...}}.
    """
    eval_dir = os.path.join(exp_dir, f"eval_{split}")
    probs = np.load(os.path.join(eval_dir, "probs.npy"))
    labels = np.load(os.path.join(eval_dir, "labels.npy"))
    n = len(labels)
    rng = np.random.default_rng(seed)

    # --- point estimate on the real (unresampled) predictions ---
    _, point_per_class, point_macro, point_overall = _compute_metrics(probs, labels, CLASS_NAMES)

    # --- metrics to track, per class and macro/overall ---
    per_class_metrics = ["sensitivity", "specificity", "ppv", "npv", "f1", "auc", "auprc"]
    summary_metrics = ["accuracy", "balanced_accuracy", "macro_f1"]

    # --- accumulate bootstrap draws ---
    boot_per_class = {name: {m: [] for m in per_class_metrics} for name in CLASS_NAMES}
    boot_macro = {m: [] for m in per_class_metrics}
    boot_overall = {m: [] for m in summary_metrics}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)   # resample with replacement, same size as the test set
        b_probs, b_labels = probs[idx], labels[idx]
        if len(np.unique(b_labels)) < NUM_CLASSES:
            continue   # skip the rare resample that drops a class entirely (undefined one-vs-rest metrics)
        _, b_per_class, b_macro, b_overall = _compute_metrics(b_probs, b_labels, CLASS_NAMES)
        for name in CLASS_NAMES:
            for m in per_class_metrics:
                boot_per_class[name][m].append(b_per_class[name][m])
        for m in per_class_metrics:
            boot_macro[m].append(b_macro[m])
        for m in summary_metrics:
            boot_overall[m].append(b_overall[m])

    # --- percentile CIs ---
    lo_pct, hi_pct = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2

    def _ci_dict(point, draws):
        draws = np.array([d for d in draws if not np.isnan(d)])   # drop undefined (e.g. PPV with no positive predictions)
        if len(draws) == 0:
            return {"point": point, "lo": None, "hi": None}
        return {"point": point, "lo": float(np.percentile(draws, lo_pct)),
                "hi": float(np.percentile(draws, hi_pct))}

    result = {
        "n_boot": n_boot, "n_resampled_out": n_boot - len(boot_overall["accuracy"]),
        "ci_level": ci, "n_test_samples": n,
        "per_class": {name: {m: _ci_dict(point_per_class[name][m], boot_per_class[name][m])
                             for m in per_class_metrics}
                     for name in CLASS_NAMES},
        "macro": {m: _ci_dict(point_macro[m], boot_macro[m]) for m in per_class_metrics},
        "overall": {m: _ci_dict(point_overall[m], boot_overall[m]) for m in summary_metrics},
    }

    out_path = os.path.join(eval_dir, "bootstrap_ci.json")
    with open(out_path, "w") as f:
        json.dump(_clean(result), f, indent=2)

    _save_ci_csv(os.path.join(eval_dir, "bootstrap_ci.csv"), CLASS_NAMES,
                 result["per_class"], result["macro"], result["overall"])

    # --- console summary ---
    print(f"\nBootstrap CIs ({n_boot} resamples, {ci:.0%} CI) on '{split}' predictions from {exp_dir}")
    print(f"{'class':<16}{'sens':>18}{'PPV':>18}{'AUC':>18}{'AUPRC':>18}")
    for name in CLASS_NAMES:
        pc = result["per_class"][name]
        def fmt(m):
            d = pc[m]
            return f"{d['point']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]" if d["lo"] is not None else "n/a"
        print(f"{name:<16}{fmt('sensitivity'):>18}{fmt('ppv'):>18}{fmt('auc'):>18}{fmt('auprc'):>18}")
    ba = result["overall"]["balanced_accuracy"]
    print(f"Balanced accuracy: {ba['point']:.4f} [{ba['lo']:.4f}, {ba['hi']:.4f}]")

    return result


if __name__ == "__main__":
    rows, median_run = summarize_runs("results/2026-09-29_15-31-21")
    bootstrap_ci(os.path.join("results/2026-09-29_15-31-21", median_run))