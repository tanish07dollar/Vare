"""
Fine-tuning script for wav2vec2
Freezes bottom 18 transformer layers, retrains top 6 + classification head
Run from: ~/
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
REAL_DIR    = Path.home() / "vare_finetune/data/real"
FAKE_DIR    = Path.home() / "vare_finetune/data/fake"
SAVE_DIR    = Path.home() / "vare_finetune/checkpoints"
LOG_PATH    = Path.home() / "vare_finetune/logs/wav2vec2_finetune.log"

MODEL_REPO  = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
SAMPLE_RATE = 16000
SEGMENT_LEN = 64600
BATCH_SIZE  = 4           # smaller batch — wav2vec2 is large
EPOCHS      = 15
LR          = 5e-5
VAL_SPLIT   = 0.15
SEED        = 42

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[wav2vec2 Finetune] Device: {device}")

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
    def __init__(self, files, labels, extractor):
        self.files     = files
        self.labels    = labels
        self.extractor = extractor

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

            waveform = audio.squeeze(0).numpy()

            inputs = self.extractor(
                waveform,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs["input_values"].squeeze(0)

        except Exception as e:
            input_values = torch.zeros(SEGMENT_LEN)

        return input_values, torch.tensor(label, dtype=torch.long)


def build_dataset(extractor):
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

    train_ds = VoiceDataset(train_files, train_labels, extractor)
    val_ds   = VoiceDataset(val_files,   val_labels,   extractor)

    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


# ─────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────
def setup_model():
    log(f"[Model] Loading {MODEL_REPO}...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_REPO)
    model     = AutoModelForAudioClassification.from_pretrained(MODEL_REPO)
    model     = model.to(device)

    # Freeze CNN feature extractor completely
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False

    # Freeze bottom 18 transformer layers (out of 24 in LARGE)
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        if i < 18:
            for param in layer.parameters():
                param.requires_grad = False

    # Everything else (top 6 layers + classifier head) stays trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log(f"[Model] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    log(f"[Model] Labels: {model.config.id2label}")

    return model, extractor


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train():
    model, extractor         = setup_model()
    train_loader, val_loader = build_dataset(extractor)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_path    = SAVE_DIR / "wav2vec2_finetuned_best.pth"

    log(f"\n[Train] Starting fine-tuning for {EPOCHS} epochs")
    log("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (input_values, labels) in enumerate(train_loader):
            input_values = input_values.to(device)
            labels       = labels.to(device)

            optimizer.zero_grad()
            output = model(input_values).logits
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

        scheduler.step()
        train_acc  = train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        val_tp, val_fp, val_tn, val_fn = 0, 0, 0, 0

        with torch.no_grad():
            for input_values, labels in val_loader:
                input_values = input_values.to(device)
                labels       = labels.to(device)
                output       = model(input_values).logits
                preds        = output.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        val_acc         = val_correct / val_total
        spoof_recall    = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        spoof_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0

        msg = (f"Epoch {epoch:02d}/{EPOCHS} | "
               f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
               f"Val Acc: {val_acc:.3f} | "
               f"Spoof Recall: {spoof_recall:.3f} | Spoof Precision: {spoof_precision:.3f}")
        log(msg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            log(f"  ✓ New best model saved (val_acc={val_acc:.3f})")

    # Also save the full model for easy loading
    full_path = SAVE_DIR / "wav2vec2_finetuned_full"
    model.save_pretrained(str(full_path))
    extractor.save_pretrained(str(full_path))
    log(f"\n[Train] Done. Best val accuracy: {best_val_acc:.3f}")
    log(f"[Train] Full model saved to: {full_path}")
    return str(full_path)


if __name__ == "__main__":
    train()
