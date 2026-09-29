import os
import json
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from dataset import ECGDataset
from params import *
from xresnet1d import xresnet1d101


def train(out_dir, fc_drop=FC_DROP, lr=LEARNING_RATE, epochs=EPOCHS,
          batch_size=BATCH_SIZE, seed=None, label_smoothing=LABEL_SMOOTHING,
          weight_decay=WEIGHT_DECAY):
    """
    Train xresnet1d101 on the splits currently in PROCESSED_DATA_DIR.

    out_dir:         folder for this experiment's results (created if missing)
    fc_drop:         dropout rate in the classifier head
    lr:              initial learning rate
    epochs:          number of training epochs (fixed, no early stopping)
    batch_size:      batch size for train and val loaders
    seed:            torch seed for weight init / dropout / batch order (None = don't seed)
    label_smoothing: smoothing used in the training loss only (0.0 = plain cross-entropy)

    Writes best_model_checkpoint.pt, last_model_checkpoint.pt and history.json
    into out_dir, and returns the history dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_set = ECGDataset('train')
    val_set = ECGDataset('val')
    train_ = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # shuffle only for training
    val_ = DataLoader(val_set, batch_size=batch_size)

    # -----------------------------------------------------------------------
    # Device and model
    # -----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xresnet1d101(in_ch=IN_CH, c_out=NUM_CLASSES, stem_k=STEM_K,
                         block_k=BLOCK_K, fc_drop=fc_drop)
    model = model.to(device)

    # -----------------------------------------------------------------------
    # Optimizer, LR scheduler, losses
    # -----------------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cut LR by 10x when val loss hasn't improved for 5 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

    # Two losses: the smoothed one is what the model optimizes; the plain one is what we
    # log and validate with, so loss curves stay comparable to runs without smoothing.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    plain_criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # Bookkeeping
    # -----------------------------------------------------------------------
    best_val_loss = float('inf')  # lowest validation loss seen so far
    best_epoch = None             # epoch (1-indexed) at which that loss occurred

    # Per-epoch history, written to out_dir/history.json.
    # "config" records this run's settings so the folder is self-describing.
    history = {
        "config": {"fc_drop": fc_drop, "lr": lr, "epochs": epochs,
                   "batch_size": batch_size, "seed": seed,
                   "label_smoothing": label_smoothing, "weight_decay": weight_decay},
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
        "lr": [],
        "best_epoch": None,
        "best_val_loss": None,
    }

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(epochs):

        # --- training phase ---
        model.train()  # enables dropout, BatchNorm uses batch statistics
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for xb, yb in train_:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()                         # clear gradients from the previous batch
            out = model(xb)                         # logits, shape (batch, NUM_CLASSES)
            loss = criterion(out, yb)               # smoothed loss drives the gradients
            loss.backward()
            opt.step()
            plain_loss = plain_criterion(out.detach(), yb)   # plain loss for logging (no extra forward pass)
            # Weight by batch size so the epoch average is exact even if the last batch is smaller
            running_train_loss += plain_loss.item() * xb.size(0)
            correct_train += (out.argmax(dim=1) == yb).sum().item()
            total_train += yb.size(0)
        avg_train_loss = running_train_loss / len(train_set)
        avg_train_acc = correct_train / total_train

        # --- validation phase ---
        model.eval()  # disables dropout, BatchNorm uses running statistics
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():  # no gradients needed for evaluation
            for xb, yb in val_:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss = plain_criterion(out, yb)   # always plain, never smoothed
                running_val_loss += val_loss.item() * xb.size(0)
                correct_val += (out.argmax(dim=1) == yb).sum().item()
                total_val += yb.size(0)
        avg_val_loss = running_val_loss / len(val_set)
        avg_val_acc = correct_val / total_val

        # --- save best checkpoint (lowest validation loss) ---
        saved = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1  # 1-indexed, matches the "Epoch n/N" printout
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model_checkpoint.pt"))
            saved = "  <- best, checkpoint saved"

        # --- learning rate tracking ---
        lr_used = opt.param_groups[0]['lr']  # LR that produced this epoch's results (read BEFORE stepping)
        scheduler.step(avg_val_loss)         # scheduler reacts to the epoch-average val loss
        lr_next = opt.param_groups[0]['lr']  # LR that will be used next epoch
        lr_note = f"  (reduced to {lr_next:.2e} for next epoch)" if lr_next < lr_used else ""

        # --- record history (rewritten every epoch so a killed run keeps its progress) ---
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)
        history["lr"].append(lr_used)
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # --- console log ---
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training loss = {avg_train_loss:.4f}, Validation loss = {avg_val_loss:.4f}{saved}")
        print(f"Training accuracy = {avg_train_acc:.4f}, Validation accuracy = {avg_val_acc:.4f}")
        print(f"Learning rate = {lr_used:.2e}{lr_note}")

    # -----------------------------------------------------------------------
    # Wrap up
    # -----------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(out_dir, "last_model_checkpoint.pt"))  # final-epoch weights
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")

    # Free memory so repeated runs in one process don't accumulate
    del model, opt, scheduler, train_, val_, train_set, val_set
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history


if __name__ == "__main__":
    train("results")  # one default run, same as the old script