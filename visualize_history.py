import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator


def plot_history(history_path, save_path=None, show=False):
    """
    Plot train/val loss and accuracy from a history.json, with a dotted line
    at the best epoch and a color strip underneath showing the learning rate.

    history_path: path to the history.json written by train.py
    save_path:    where to save the figure (skipped if None)
    show:         whether to display the figure interactively
    """
    # --- load history ---
    with open(history_path) as f:
        h = json.load(f)

    n = len(h["train_loss"])
    epochs = np.arange(1, n + 1)   # 1-indexed, matches best_epoch in the JSON
    lr = np.array(h["lr"])

    # --- figure: main plot on top, thin LR strip underneath (shared epoch axis) ---
    fig, (ax, ax_lr) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 6),
        gridspec_kw={"height_ratios": [12, 1], "hspace": 0.08},
    )
    ax2 = ax.twinx()   # right-hand axis for accuracy

    # color = train vs val, linestyle = loss (solid) vs accuracy (dashed)
    c_train, c_val = "tab:blue", "tab:orange"
    ax.plot(epochs, h["train_loss"], color=c_train, ls="-", label="Train loss")
    ax.plot(epochs, h["val_loss"], color=c_val, ls="-", label="Val loss")
    ax2.plot(epochs, h["train_acc"], color=c_train, ls="--", label="Train accuracy")
    ax2.plot(epochs, h["val_acc"], color=c_val, ls="--", label="Val accuracy")

    # dotted vertical line at the best epoch (dotted is reserved for this marker)
    if h.get("best_epoch") is not None:
        ax.axvline(h["best_epoch"], color="gray", ls=":", lw=1.5,
                   label=f"Best epoch ({h['best_epoch']})")

    ax.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, n + 0.5)

    # one combined legend, placed on ax2 so the accuracy lines don't draw over it
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="center right", framealpha=0.9)

    if h.get("best_val_loss") is not None:
        ax.set_title("Training history")

    # --- LR strip: one colored cell per epoch, log color scale ---
    vmin, vmax = lr.min(), lr.max()
    if vmin == vmax:          # LR never changed: LogNorm needs a nonzero range
        vmin = vmin / 10
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    x_edges = np.arange(0.5, n + 1.5, 1.0)   # each cell centered on its epoch
    ax_lr.pcolormesh(x_edges, [0, 1], lr[None, :], cmap=cmap, norm=norm)

    # label each constant-LR segment directly on the strip (replaces a colorbar)
    start = 0
    for i in range(1, n + 1):
        if i == n or lr[i] != lr[i - 1]:
            center = (start + 1 + i) / 2      # segment covers epochs start+1 .. i
            r, g, b, _ = cmap(norm(lr[start]))
            text_color = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
            ax_lr.text(center, 0.5, f"{lr[start]:.0e}", ha="center", va="center",
                       fontsize=8, color=text_color)
            start = i

    ax_lr.set_yticks([])
    ax_lr.set_ylabel("LR", rotation=0, ha="right", va="center")
    ax_lr.set_xlabel("Epoch")
    ax_lr.xaxis.set_major_locator(MaxNLocator(integer=True))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)   # free memory when running many experiments

    return fig


if __name__ == "__main__":
    plot_history("results/history.json", "results/history_plot.png")
    plot_history("results/history.json", "results/history_plot_2.png")