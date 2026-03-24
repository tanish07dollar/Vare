"""
RawNet2 training from scratch on VARE dataset
Uses RawNet2Spoof.py from the aasist repo
Run from: ~/
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path.home() / "aasist"))
from models.RawNet2Spoof import Model

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
REAL_DIR    = Path.home() / "vare_finetune/data/real"
FAKE_DIR    = Path.home() / "vare_finetune/data/fake"
SAVE_DIR    = Path.home() / "vare_finetune/checkpoints"
LOG_PATH    = Path.home() / "vare_finetune/logs/rawnet2_train.log"

SAMPLE_RATE = 16000
SEGMENT_LEN = 64600
BATCH_SIZE  = 16          # RawNet2 is lightweight — can use larger batch
EPOCHS      = 30
LR          = 1e-3
VAL_SPLIT   = 0.15
SEED        = 42

# RawNet2 model config — matches ASVspoof baseline
MODEL_CONFIG = {
    "nb_samp":    64600,
    "first_conv": 1024,
    "in_channels": 1,
    "filts":      [20, [20, 20], [20, 128], [128, 128]],
    "blocks":     [2, 4],
    "nb_fc_node": 1024,
    "gru_node":   1024,
    "nb_gru_layer": 3,
    "nb_classes": 2
}

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[RawNet2 Train] Device: {device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def log(msg):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class VoiceDataset(Dataset):
    def __init__(self, files, labels):
        self.files  = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        label = self.labels[idx]

        try:
            audio, sr = torchaudio.load(str(path))
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)

            if audio.shape[1] < SEGMENT_LEN:
                audio = torch.nn.functional.pad(audio, (0, SEGMENT_LEN - audio.shape[1]))
            else:
                start = random.randint(0, audio.shape[1] - SEGMENT_LEN)
                audio = audio[:, start:start + SEGMENT_LEN]

        except Exception:
            audio = torch.zeros(1, SEGMENT_LEN)

        return audio.squeeze(0), torch.tensor(label, dtype=torch.long)


def build_dataset():
    real_files = list(REAL_DIR.glob("*.wav"))
    fake_files = list(FAKE_DIR.glob("*.wav"))

    log(f"[Data] Real: {len(real_files)}  Fake: {len(fake_files)}")

    all_files  = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)

    split = int(len(all_files) * (1 - VAL_SPLIT))
    train_files,  val_files  = all_files[:split],  all_files[split:]
    train_labels, val_labels = all_labels[:split], all_labels[split:]

    log(f"[Data] Train: {len(train_files)}  Val: {len(val_files)}")

    train_ds = VoiceDataset(train_files, train_labels)
    val_ds   = VoiceDataset(val_files,   val_labels)

    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train():
    train_loader, val_loader = build_dataset()

    log("[Model] Building RawNet2 from scratch...")
    model = Model(MODEL_CONFIG).to(device)

    total = sum(p.numel() for p in model.parameters())
    log(f"[Model] Total parameters: {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, verbose=True
    )

    best_val_acc = 0.0
    best_path    = SAVE_DIR / "rawnet2_trained_best.pth"

    log(f"\n[Train] Starting training for {EPOCHS} epochs")
    log("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (audio, labels) in enumerate(train_loader):
            audio  = audio.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(audio)[1]
            loss   = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item()
            preds          = output.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", end="\r")

        train_acc  = train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        val_tp, val_fp, val_tn, val_fn = 0, 0, 0, 0

        with torch.no_grad():
            for audio, labels in val_loader:
                audio  = audio.to(device)
                labels = labels.to(device)
                output = model(audio)[1]
                preds  = output.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        val_acc         = val_correct / val_total
        spoof_recall    = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        spoof_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0

        scheduler.step(val_acc)

        msg = (f"Epoch {epoch:02d}/{EPOCHS} | "
               f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
               f"Val Acc: {val_acc:.3f} | "
               f"Spoof Recall: {spoof_recall:.3f} | Spoof Precision: {spoof_precision:.3f}")
        log(msg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "model_config": {"nb_samp": 64600, "first_conv": 1024, "in_channels": 1, "filts": [20, [20, 20], [20, 128], [128, 128]], "blocks": [2, 4], "nb_fc_node": 1024, "gru_node": 1024, "nb_gru_layer": 3, "nb_classes": 2},
                "val_acc":      val_acc,
            }, best_path)
            log(f"  ✓ New best model saved (val_acc={val_acc:.3f})")

    log(f"\n[Train] Done. Best val accuracy: {best_val_acc:.3f}")
    log(f"[Train] Best model saved to: {best_path}")
    return str(best_path)


if __name__ == "__main__":
    train()
