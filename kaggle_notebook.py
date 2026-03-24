# =============================================================================
# VARE — Kaggle Training Notebook
# Paste each cell block into a separate Kaggle notebook cell.
# GPU: T4 (15 GB VRAM) | Storage: /kaggle/working/ (20 GB)
# =============================================================================

# =============================================================================
# CELL 1 — GPU Check
# =============================================================================
import torch, os

print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch        : {torch.__version__}")

# =============================================================================
# CELL 2 — Install Dependencies
# =============================================================================
# Run this cell first — it installs packages not present by default on Kaggle.
# Takes ~2 minutes.

import subprocess
subprocess.run(["pip", "install", "-q", "transformers==4.40.0", "accelerate"], check=True)

# torchaudio, numpy, scipy are already available on Kaggle.

# =============================================================================
# CELL 3 — Clone VARE Repo from GitHub
# =============================================================================
# BEFORE running this cell:
#   1. Push your local VARE project to a GitHub repo (private is fine).
#   2. In Kaggle → Add-ons → Secrets → add a secret named GITHUB_TOKEN
#      with a GitHub personal access token (classic, repo scope).
#
# If you don't want to use GitHub, skip this cell and instead upload your
# model files as a Kaggle Dataset, then adjust VARE_DIR below.

from kaggle_secrets import UserSecretsClient
import subprocess, os

secret = UserSecretsClient()
token  = secret.get_secret("GITHUB_TOKEN")

GITHUB_USER = "YOUR_GITHUB_USERNAME"   # <-- change this
GITHUB_REPO = "VARE"                   # <-- change this if different

repo_url = f"https://{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

if not os.path.exists("/kaggle/working/VARE"):
    subprocess.run(["git", "clone", repo_url, "/kaggle/working/VARE"], check=True)
    print("Repo cloned successfully.")
else:
    subprocess.run(["git", "-C", "/kaggle/working/VARE", "pull"], check=True)
    print("Repo updated.")

VARE_DIR = "/kaggle/working/VARE"

# =============================================================================
# CELL 4 — Verify ASVspoof 2019 LA Dataset
# =============================================================================
# BEFORE running this cell, upload ASVspoof 2019 LA to Kaggle:
#   1. Download from: https://datashare.ed.ac.uk/handle/10283/3336
#      (Free. Requires a free University of Edinburgh login.)
#      Download "LA.zip" (~14 GB)
#   2. In Kaggle → Datasets → New Dataset
#      Name it exactly: asvspoof2019-la
#      Upload LA.zip → Kaggle extracts it automatically.
#   3. In this notebook → Add Data → search "asvspoof2019-la" → Add it.
#      It will appear at /kaggle/input/asvspoof2019-la/

from pathlib import Path

DATASET_ROOT = Path("/kaggle/input/asvspoof2019-la/LA")

TRAIN_AUDIO = DATASET_ROOT / "ASVspoof2019_LA_train/flac"
DEV_AUDIO   = DATASET_ROOT / "ASVspoof2019_LA_dev/flac"
EVAL_AUDIO  = DATASET_ROOT / "ASVspoof2019_LA_eval/flac"

TRAIN_PROTO = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTO   = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTO  = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

# Verify everything is in place
for p in [TRAIN_AUDIO, DEV_AUDIO, EVAL_AUDIO, TRAIN_PROTO, DEV_PROTO, EVAL_PROTO]:
    status = "OK" if p.exists() else "MISSING"
    print(f"[{status}] {p}")

train_count = len(list(TRAIN_AUDIO.glob("*.flac")))
dev_count   = len(list(DEV_AUDIO.glob("*.flac")))
eval_count  = len(list(EVAL_AUDIO.glob("*.flac")))
print(f"\nTrain FLAC: {train_count}")
print(f"Dev   FLAC: {dev_count}")
print(f"Eval  FLAC: {eval_count}")

# Expected:  Train ~25,380 | Dev ~24,844 | Eval ~71,237

# =============================================================================
# CELL 5 — Shared Utilities (run before any training cell)
# =============================================================================
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

SEED        = 42
SAMPLE_RATE = 16000
SEGMENT_LEN = 64600   # ~4 seconds at 16kHz

OUTPUT_DIR  = Path("/kaggle/working/checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_protocol(protocol_path, audio_dir):
    """
    Parse an ASVspoof 2019 LA protocol file.
    Protocol format (space-separated):
        speaker_id  file_id  -  system_id  label
        LA_0079     LA_T_1138215  -  -   bonafide
        LA_0013     LA_T_1271478  -  A01 spoof
    Returns: (file_paths, labels) where label 0=real, 1=fake
    """
    files, labels = [], []
    missing = 0
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id   = parts[1]
            label_str = parts[4]
            label     = 0 if label_str == "bonafide" else 1
            path      = Path(audio_dir) / f"{file_id}.flac"
            if path.exists():
                files.append(path)
                labels.append(label)
            else:
                missing += 1

    real_count = labels.count(0)
    fake_count = labels.count(1)
    print(f"  Loaded {len(files)} files | Real: {real_count} | Fake: {fake_count} | Missing: {missing}")
    return files, labels


def load_audio(path):
    """Load and preprocess a single audio file to a fixed-length tensor."""
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
    return audio.squeeze(0)


class ASVspoofDataset(Dataset):
    def __init__(self, files, labels):
        self.files  = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = load_audio(self.files[idx])
        return audio, torch.tensor(self.labels[idx], dtype=torch.long)


def make_loaders(train_files, train_labels, dev_files, dev_labels,
                 batch_size, num_workers=2):
    train_ds = ASVspoofDataset(train_files, train_labels)
    dev_ds   = ASVspoofDataset(dev_files,   dev_labels)

    # Weighted sampler to handle class imbalance (ASVspoof has ~10x more fakes)
    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights      = [1.0 / class_counts[l] for l in train_labels]
    sampler      = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, dev_loader


def log_print(msg, log_path):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


def eval_epoch(model, loader, forward_fn):
    """Shared validation loop. forward_fn(model, batch) → logits."""
    model.eval()
    correct, total = 0, 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for audio, labels in loader:
            audio, labels = audio.to(device), labels.to(device)
            logits = forward_fn(model, audio)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    acc       = correct / total
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return acc, recall, precision


print("Utilities loaded.")

# =============================================================================
# CELL 6 — Train RawNet2
# =============================================================================
import sys
sys.path.insert(0, f"{VARE_DIR}/models/aasist_reference")
from models.RawNet2Spoof import Model as RawNet2

LOG_PATH = OUTPUT_DIR / "rawnet2_train.log"

MODEL_CONFIG = {
    "nb_samp":      64600,
    "first_conv":   1024,
    "in_channels":  1,
    "filts":        [20, [20, 20], [20, 128], [128, 128]],
    "blocks":       [2, 4],
    "nb_fc_node":   1024,
    "gru_node":     1024,
    "nb_gru_layer": 3,
    "nb_classes":   2,
}

BATCH_SIZE = 24
EPOCHS     = 30
LR         = 1e-3

def log(msg): log_print(msg, LOG_PATH)

log("=== RawNet2 Training ===")
log("Loading ASVspoof 2019 LA train/dev splits...")

train_files, train_labels = load_protocol(TRAIN_PROTO, TRAIN_AUDIO)
dev_files,   dev_labels   = load_protocol(DEV_PROTO,   DEV_AUDIO)

train_loader, dev_loader = make_loaders(
    train_files, train_labels, dev_files, dev_labels,
    batch_size=BATCH_SIZE, num_workers=4
)

model     = RawNet2(MODEL_CONFIG).to(device)
total_p   = sum(p.numel() for p in model.parameters())
log(f"Parameters: {total_p:,}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=4
)

best_acc  = 0.0
best_path = OUTPUT_DIR / "rawnet2_best.pth"

def rawnet2_forward(m, audio):
    return m(audio)[1]

log(f"Training for {EPOCHS} epochs on {device}")
log("=" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = train_correct = train_total = 0

    for i, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = rawnet2_forward(model, audio)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

        if i % 50 == 0:
            print(f"  Ep {epoch} | Batch {i}/{len(train_loader)} | Loss {loss.item():.4f}", end="\r")

    train_acc  = train_correct / train_total
    train_loss = train_loss / len(train_loader)

    val_acc, val_recall, val_prec = eval_epoch(model, dev_loader, rawnet2_forward)
    scheduler.step(val_acc)

    msg = (f"Epoch {epoch:02d}/{EPOCHS} | "
           f"Loss {train_loss:.4f} | TrainAcc {train_acc:.3f} | "
           f"ValAcc {val_acc:.3f} | Recall {val_recall:.3f} | Prec {val_prec:.3f}")
    log(msg)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "model_config": MODEL_CONFIG,
            "val_acc":      val_acc,
        }, best_path)
        log(f"  ✓ Saved best (val_acc={val_acc:.3f})")

log(f"\nDone. Best val_acc: {best_acc:.3f}")
log(f"Checkpoint: {best_path}")

# =============================================================================
# CELL 7 — Fine-tune wav2vec2
# =============================================================================
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import gc

LOG_PATH_W2V = OUTPUT_DIR / "wav2vec2_finetune.log"

MODEL_REPO  = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
BATCH_SIZE  = 4    # wav2vec2-large needs small batches on T4
EPOCHS      = 15
LR          = 5e-5

def logw(msg): log_print(msg, LOG_PATH_W2V)

logw("=== wav2vec2 Fine-tuning ===")
logw("Loading ASVspoof 2019 LA train/dev splits...")

train_files_w, train_labels_w = load_protocol(TRAIN_PROTO, TRAIN_AUDIO)
dev_files_w,   dev_labels_w   = load_protocol(DEV_PROTO,   DEV_AUDIO)

# wav2vec2 needs a feature extractor — wrap the dataset
class W2VDataset(Dataset):
    def __init__(self, files, labels, extractor):
        self.files     = files
        self.labels    = labels
        self.extractor = extractor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = load_audio(self.files[idx]).numpy()
        inputs = self.extractor(
            audio, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True
        )
        return inputs["input_values"].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)


logw(f"Loading model: {MODEL_REPO}")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_REPO)
w2v_model = AutoModelForAudioClassification.from_pretrained(MODEL_REPO).to(device)

# Freeze CNN feature extractor
for param in w2v_model.wav2vec2.feature_extractor.parameters():
    param.requires_grad = False

# Freeze bottom 18 of 24 transformer layers
for i, layer in enumerate(w2v_model.wav2vec2.encoder.layers):
    if i < 18:
        for param in layer.parameters():
            param.requires_grad = False

trainable = sum(p.numel() for p in w2v_model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in w2v_model.parameters())
logw(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

# Build dataloaders with W2VDataset
train_ds_w = W2VDataset(train_files_w, train_labels_w, extractor)
dev_ds_w   = W2VDataset(dev_files_w,   dev_labels_w,   extractor)

class_counts = [train_labels_w.count(0), train_labels_w.count(1)]
weights      = [1.0 / class_counts[l] for l in train_labels_w]
sampler_w    = WeightedRandomSampler(weights, num_samples=len(train_labels_w), replacement=True)

train_loader_w = DataLoader(train_ds_w, batch_size=BATCH_SIZE, sampler=sampler_w, num_workers=2, pin_memory=True)
dev_loader_w   = DataLoader(dev_ds_w,   batch_size=BATCH_SIZE, shuffle=False,     num_workers=2, pin_memory=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, w2v_model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc_w  = 0.0
best_path_w = OUTPUT_DIR / "wav2vec2_best.pth"
full_path_w = OUTPUT_DIR / "wav2vec2_full"

def w2v_forward(m, audio):
    return m(audio).logits

logw(f"Fine-tuning for {EPOCHS} epochs on {device}")
logw("=" * 60)

for epoch in range(1, EPOCHS + 1):
    w2v_model.train()
    train_loss = train_correct = train_total = 0

    for i, (inputs, labels) in enumerate(train_loader_w):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = w2v_forward(w2v_model, inputs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(w2v_model.parameters(), 1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

        if i % 50 == 0:
            print(f"  Ep {epoch} | Batch {i}/{len(train_loader_w)} | Loss {loss.item():.4f}", end="\r")

    scheduler.step()
    train_acc  = train_correct / train_total
    train_loss = train_loss / len(train_loader_w)

    val_acc, val_recall, val_prec = eval_epoch(w2v_model, dev_loader_w, w2v_forward)

    msg = (f"Epoch {epoch:02d}/{EPOCHS} | "
           f"Loss {train_loss:.4f} | TrainAcc {train_acc:.3f} | "
           f"ValAcc {val_acc:.3f} | Recall {val_recall:.3f} | Prec {val_prec:.3f}")
    logw(msg)

    if val_acc > best_acc_w:
        best_acc_w = val_acc
        torch.save(w2v_model.state_dict(), best_path_w)
        logw(f"  ✓ Saved best (val_acc={val_acc:.3f})")

# Save full model + extractor for easy reloading in app.py
w2v_model.save_pretrained(str(full_path_w))
extractor.save_pretrained(str(full_path_w))

logw(f"\nDone. Best val_acc: {best_acc_w:.3f}")
logw(f"Full model: {full_path_w}")

# Free VRAM before next model
del w2v_model, train_ds_w, dev_ds_w
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# CELL 8 — Fine-tune AASIST3
# =============================================================================
import sys
sys.path.insert(0, f"{VARE_DIR}/models/aasist3")
from model import aasist3

LOG_PATH_A3 = OUTPUT_DIR / "aasist3_finetune.log"

BATCH_SIZE = 8
EPOCHS     = 20
LR         = 1e-4

def loga(msg): log_print(msg, LOG_PATH_A3)

loga("=== AASIST3 Fine-tuning ===")
loga("Loading ASVspoof 2019 LA train/dev splits...")

train_files_a, train_labels_a = load_protocol(TRAIN_PROTO, TRAIN_AUDIO)
dev_files_a,   dev_labels_a   = load_protocol(DEV_PROTO,   DEV_AUDIO)

train_loader_a, dev_loader_a = make_loaders(
    train_files_a, train_labels_a, dev_files_a, dev_labels_a,
    batch_size=BATCH_SIZE, num_workers=4
)

loga("Loading AASIST3 pretrained weights from MTUCI/AASIST3...")
a3_model = aasist3.from_pretrained("MTUCI/AASIST3").to(device)
torch.cuda.empty_cache()

# Freeze all parameters first
for param in a3_model.parameters():
    param.requires_grad = False

# Unfreeze classification head layers only
unfrozen = 0
for name, param in a3_model.named_parameters():
    if any(k in name for k in ["out_layer", "master", "fc", "classifier", "head", "branch"]):
        param.requires_grad = True
        unfrozen += 1

# Fallback: unfreeze last 30% if head names not found
if unfrozen == 0:
    loga("Head layers not found by name — unfreezing last 30% of parameters")
    all_params = list(a3_model.parameters())
    cutoff     = int(len(all_params) * 0.7)
    for param in all_params[cutoff:]:
        param.requires_grad = True
    unfrozen = len(all_params) - cutoff

trainable = sum(p.numel() for p in a3_model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in a3_model.parameters())
loga(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, a3_model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc_a  = 0.0
best_path_a = OUTPUT_DIR / "aasist3_best.pth"

def aasist3_forward(m, audio):
    return m(audio)

loga(f"Fine-tuning for {EPOCHS} epochs on {device}")
loga("=" * 60)

for epoch in range(1, EPOCHS + 1):
    a3_model.train()
    train_loss = train_correct = train_total = 0

    for i, (audio, labels) in enumerate(train_loader_a):
        audio, labels = audio.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = aasist3_forward(a3_model, audio)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a3_model.parameters(), 1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

        if i % 50 == 0:
            print(f"  Ep {epoch} | Batch {i}/{len(train_loader_a)} | Loss {loss.item():.4f}", end="\r")

    scheduler.step()
    train_acc  = train_correct / train_total
    train_loss = train_loss / len(train_loader_a)

    val_acc, val_recall, val_prec = eval_epoch(a3_model, dev_loader_a, aasist3_forward)

    msg = (f"Epoch {epoch:02d}/{EPOCHS} | "
           f"Loss {train_loss:.4f} | TrainAcc {train_acc:.3f} | "
           f"ValAcc {val_acc:.3f} | Recall {val_recall:.3f} | Prec {val_prec:.3f}")
    loga(msg)

    if val_acc > best_acc_a:
        best_acc_a = val_acc
        torch.save(a3_model.state_dict(), best_path_a)
        loga(f"  ✓ Saved best (val_acc={val_acc:.3f})")

loga(f"\nDone. Best val_acc: {best_acc_a:.3f}")
loga(f"Checkpoint: {best_path_a}")

# =============================================================================
# CELL 9 — Summary & Download Checkpoints
# =============================================================================
import os

print("\n=== Training Complete ===\n")
print(f"{'Checkpoint':<40} {'Size':>10}")
print("-" * 52)
for f in sorted(OUTPUT_DIR.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1e6
        print(f"{str(f.relative_to(OUTPUT_DIR)):<40} {size_mb:>9.1f} MB")

print("\n--- Results Summary ---")
print(f"RawNet2  best val_acc : {best_acc:.3f}")
print(f"wav2vec2 best val_acc : {best_acc_w:.3f}")
print(f"AASIST3  best val_acc : {best_acc_a:.3f}")

print("\nAll checkpoints are in /kaggle/working/checkpoints/")
print("Download them via Kaggle UI: Output tab → Download All")

# =============================================================================
# CELL 10 — (Optional) Zip & Download in One File
# =============================================================================
import shutil

zip_path = "/kaggle/working/vare_checkpoints"
shutil.make_archive(zip_path, "zip", OUTPUT_DIR)
print(f"Zipped to: {zip_path}.zip")
print(f"Size: {os.path.getsize(zip_path + '.zip') / 1e6:.1f} MB")
print("Download from Kaggle Output tab.")
