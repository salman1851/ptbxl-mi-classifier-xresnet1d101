import os
import json
import traceback
from datetime import datetime

from params import *
from preprocessing import preprocess
from train import train
from visualize_history import plot_history
from evaluate import evaluate


def run_experiments():
    # one time-stamped folder per invocation, one subfolder per experiment
    run_dir = os.path.join("results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results folder: {run_dir}")

    for i in range(N_EXPERIMENTS):
        seed = RANDOM_SEED + i                      # different seed = different split and init
        out_dir = os.path.join(run_dir, f"exp_{i + 1}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n===== Experiment {i + 1}/{N_EXPERIMENTS} (seed {seed}) =====")

        try:
            print("Preprocessing...")
            counts = preprocess(seed=seed)          # fresh split, overwrites PROCESSED_DATA_DIR
            with open(os.path.join(out_dir, "split_counts.json"), "w") as f:
                json.dump({"seed": seed, "counts": counts}, f, indent=2)

            print("Training...")
            train(out_dir, seed=seed)               # all other settings come from params.py
            print("Logging history...")
            plot_history(os.path.join(out_dir, "history.json"),
                         os.path.join(out_dir, "history_plot.png"))
            print("Evaluating...")
            evaluate(out_dir, split="val")          # val for comparing runs
            evaluate(out_dir, split="test")         # test for the final report

        except Exception:                           # log the failure, carry on with the rest
            traceback.print_exc()
            with open(os.path.join(out_dir, "FAILED.txt"), "w") as f:
                f.write(traceback.format_exc())


if __name__ == "__main__":
    run_experiments()