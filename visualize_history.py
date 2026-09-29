import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator


def _draw_lr_strip(ax_lr, epochs, lr):
    """Shared LR strip: one colored cell per epoch, log color scale, labeled segments."""
    n = len(epochs)
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


def _new_figure(n):
    """Main plot on top, thin LR strip underneath, sharing the epoch axis."""
    fig, (ax, ax_lr) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5.5),
        gridspec_kw={"height_ratios": [12, 1], "hspace": 0.08},
    )
    ax.set_xlim(0.5, n + 0.5)
    return fig, ax, ax_lr


def plot_loss_history(history_path, save_path=None, show=False):
    """
    Plot train/val loss from a history.json, with a dotted line at the best
    epoch and a color strip underneath showing the learning rate. Y-axis is
    fixed to [0, 1].
    """
    with open(history_path) as f:
        h = json.load(f)

    n = len(h["train_loss"])
    epochs = np.arange(1, n + 1)   # 1-indexed, matches best_epoch in the JSON
    lr = np.array(h["lr"])

    fig, ax, ax_lr = _new_figure(n)
    ax.plot(epochs, h["train_loss"], color="tab:blue", label="Train loss")
    ax.plot(epochs, h["val_loss"], color="tab:orange", label="Val loss")
    if h.get("best_epoch") is not None:
        ax.axvline(h["best_epoch"], color="gray", ls=":", lw=1.5,
                   label=f"Best epoch ({h['best_epoch']})")

    ax.set_ylabel("Loss")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title("Training history — loss")

    _draw_lr_strip(ax_lr, epochs, lr)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_accuracy_history(history_path, save_path=None, show=False):
    """
    Plot train/val accuracy from a history.json, with a dotted line at the
    best epoch (by validation loss) and a color strip underneath showing the
    learning rate. Y-axis is fixed to [0, 1].
    """
    with open(history_path) as f:
        h = json.load(f)

    n = len(h["train_acc"])
    epochs = np.arange(1, n + 1)
    lr = np.array(h["lr"])

    fig, ax, ax_lr = _new_figure(n)
    ax.plot(epochs, h["train_acc"], color="tab:blue", label="Train accuracy")
    ax.plot(epochs, h["val_acc"], color="tab:orange", label="Val accuracy")
    if h.get("best_epoch") is not None:
        ax.axvline(h["best_epoch"], color="gray", ls=":", lw=1.5,
                   label=f"Best epoch ({h['best_epoch']})")

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title("Training history — accuracy")

    _draw_lr_strip(ax_lr, epochs, lr)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


if __name__ == "__main__":
    plot_loss_history("results/history.json", "results/loss_plot.png")
    plot_accuracy_history("results/history.json", "results/accuracy_plot.png")