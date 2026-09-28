import os
import json
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from dataset import ECGDataset
from params import *
from xresnet1d import xresnet1d101


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_set = ECGDataset('train')
val_set = ECGDataset('val')
train_ = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # shuffle only for training
val_ = DataLoader(val_set, batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------
# Device and model
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = xresnet1d101(in_ch=IN_CH, c_out=NUM_CLASSES, stem_k=STEM_K,
                     block_k=BLOCK_K, fc_drop=0.25)  # dropout in the classifier head only
model = model.to(device)

# ---------------------------------------------------------------------------
# Optimizer, LR scheduler, loss
# ---------------------------------------------------------------------------
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # weight_decay left at the 0.01 default
# Cut LR by 10x when val loss hasn't improved for 5 consecutive epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)
criterion = nn.CrossEntropyLoss()  # expects raw logits + integer class targets

# ---------------------------------------------------------------------------
# Output folder and bookkeeping
# ---------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

best_val_loss = float('inf')  # lowest validation loss seen so far
best_epoch = None             # epoch (1-indexed) at which that loss occurred

# Per-epoch history, written to results/history.json
history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [],   "val_acc": [],
    "lr": [],
    "best_epoch": None,
    "best_val_loss": None,
}

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for epoch in range(EPOCHS):

    # --- training phase ---
    model.train()  # enables dropout, BatchNorm uses batch statistics
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for xb, yb in train_:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()                         # clear gradients from the previous batch
        out = model(xb)                         # logits, shape (batch, NUM_CLASSES)
        train_loss = criterion(out, yb)
        train_loss.backward()
        opt.step()
        # Weight by batch size so the epoch average is exact even if the last batch is smaller
        running_train_loss += train_loss.item() * xb.size(0)
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
            val_loss = criterion(out, yb)
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
        torch.save(model.state_dict(), "results/best_model_checkpoint.pt")
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
    with open("results/history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- console log ---
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Training loss = {avg_train_loss:.4f}, Validation loss = {avg_val_loss:.4f}{saved}")
    print(f"Training accuracy = {avg_train_acc:.4f}, Validation accuracy = {avg_val_acc:.4f}")
    print(f"Learning rate = {lr_used:.2e}{lr_note}")

# ---------------------------------------------------------------------------
# Wrap up
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "results/last_model_checkpoint.pt")  # final-epoch weights, for comparison
print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")