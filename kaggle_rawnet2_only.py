# =============================================================================
# VARE — RawNet2 Only Training Notebook
# Paste each CELL block into a separate Kaggle notebook cell.
# GPU required: T4 (15 GB VRAM)
# Total expected runtime: ~45-60 minutes
# =============================================================================

# =============================================================================
# CELL 1 — GPU & Environment Check
# =============================================================================
import torch, os, sys

print("=" * 50)
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM            : {vram:.1f} GB")
print(f"PyTorch         : {torch.__version__}")
print(f"Python          : {sys.version.split()[0]}")
print("=" * 50)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Go to Settings → Accelerator → GPU T4 x1")

# =============================================================================
# CELL 2 — Install Dependencies
# =============================================================================
import subprocess

packages = [
    "transformers>=4.41.0",
    "kokoro>=0.9.2",
    "edge-tts",
    "nest_asyncio",
    "soundfile",
    "scipy",
]

for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.run(["pip", "install", "-q", pkg], check=True)

print("\nAll packages installed.")

# =============================================================================
# CELL 3 — Clone VARE from GitHub
# =============================================================================
# SETUP REQUIRED:
#   Kaggle → Add-ons → Secrets → Add secret:
#     Name:  GITHUB_TOKEN
#     Value: your GitHub personal access token (repo scope)
#     Toggle "Notebook access" ON

from kaggle_secrets import UserSecretsClient
import subprocess, os, sys

GITHUB_USER = "tanish07dollar"   # your GitHub username
GITHUB_REPO = "Vare"

token    = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_url = f"https://{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
vare_dir = "/kaggle/working/VARE"

if os.path.exists(vare_dir):
    import shutil
    shutil.rmtree(vare_dir)

subprocess.run(["git", "clone", repo_url, vare_dir], check=True)
print("Repo cloned.")

sys.path.insert(0, vare_dir)
sys.path.insert(0, f"{vare_dir}/models/aasist_reference")
print(f"VARE path added: {vare_dir}")

# =============================================================================
# CELL 4 — Download LibriSpeech (real audio) + Generate Fake Audio
# =============================================================================
# LibriSpeech dev-clean: ~337 MB, ~2700 clips — downloads in ~2 min
# Fake generation: 500 Kokoro + 500 edge-tts — ~15-20 min

import os, asyncio, soundfile as sf
import torchaudio
from pathlib import Path

REAL_DIR = Path("/kaggle/working/real_audio")
FAKE_DIR = Path("/kaggle/working/fake_audio")
REAL_DIR.mkdir(exist_ok=True)
FAKE_DIR.mkdir(exist_ok=True)

# ── Download LibriSpeech dev-clean ──
print("Downloading LibriSpeech dev-clean...")
dataset = torchaudio.datasets.LIBRISPEECH(
    root="/kaggle/working",
    url="dev-clean",
    download=True,
)
print(f"LibriSpeech loaded: {len(dataset)} clips")

# Convert to .wav files in REAL_DIR
print("Exporting real audio to wav...")
for i, (waveform, sr, _, _, _, _) in enumerate(dataset):
    out = REAL_DIR / f"real_{i:05d}.wav"
    if not out.exists():
        torchaudio.save(str(out), waveform, sr)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(dataset)} exported", end="\r")
print(f"\nReal audio ready: {len(list(REAL_DIR.glob('*.wav')))} files")

# ── Generate fake audio — Kokoro ──
N_FAKE = 500   # per system — enough for good training, fast to generate

FALLBACK_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "To be or not to be, that is the question whether tis nobler in the mind.",
    "I wandered lonely as a cloud that floats on high over vales and hills.",
    "All that glitters is not gold, often have you heard that told.",
    "It was the best of times it was the worst of times it was the age of wisdom.",
    "Call me Ishmael. Some years ago, never mind how long precisely, I thought I would sail.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune.",
] * 60   # repeat to get 600 sentences

kokoro_dir = FAKE_DIR / "kokoro"
kokoro_dir.mkdir(exist_ok=True)
already_done = len(list(kokoro_dir.glob("*.wav")))

if already_done >= N_FAKE:
    print(f"Kokoro: {already_done} files already done, skipping.")
else:
    print(f"\nGenerating {N_FAKE} Kokoro clips...")
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")
        voices   = ["af_heart", "af_sky", "am_adam"]
        generated = already_done

        for i, text in enumerate(FALLBACK_TEXTS[already_done:N_FAKE], start=already_done):
            voice = voices[i % len(voices)]
            try:
                for _, _, audio in pipeline(text, voice=voice, speed=1.0):
                    out = kokoro_dir / f"kokoro_{i:04d}.wav"
                    sf.write(str(out), audio, 24000)
                    generated += 1
                    break
            except Exception:
                continue
            if (i + 1) % 50 == 0:
                print(f"  Kokoro: {generated}/{N_FAKE}", end="\r")

        print(f"\nKokoro done: {generated} files")
    except Exception as e:
        print(f"Kokoro failed: {e}")

# ── Generate fake audio — edge-tts ──
edgetts_dir = FAKE_DIR / "edge_tts"
edgetts_dir.mkdir(exist_ok=True)
already_done_e = len(list(edgetts_dir.glob("*.wav")))

if already_done_e >= N_FAKE:
    print(f"edge-tts: {already_done_e} files already done, skipping.")
else:
    print(f"\nGenerating {N_FAKE} edge-tts clips...")
    try:
        import nest_asyncio, edge_tts
        nest_asyncio.apply()

        EDGE_VOICES = [
            "en-US-JennyNeural",
            "en-US-GuyNeural",
            "en-GB-SoniaNeural",
            "en-AU-NatashaNeural",
        ]

        async def gen_one(text, voice, out_path):
            try:
                tmp = str(out_path).replace(".wav", ".mp3")
                await edge_tts.Communicate(text, voice).save(tmp)
                w, sr = torchaudio.load(tmp)
                if w.shape[0] > 1:
                    w = w.mean(0, keepdim=True)
                torchaudio.save(str(out_path), w, sr)
                os.remove(tmp)
                return True
            except Exception:
                return False

        async def run_edge():
            total = already_done_e
            texts = FALLBACK_TEXTS[already_done_e:N_FAKE]
            for i in range(0, len(texts), 10):
                batch   = texts[i:i+10]
                tasks   = [
                    gen_one(t, EDGE_VOICES[(already_done_e+i+j) % len(EDGE_VOICES)],
                            edgetts_dir / f"edgetts_{already_done_e+i+j:04d}.wav")
                    for j, t in enumerate(batch)
                    if not (edgetts_dir / f"edgetts_{already_done_e+i+j:04d}.wav").exists()
                ]
                results = await asyncio.gather(*tasks)
                total  += sum(results)
                print(f"  edge-tts: {total}/{N_FAKE}", end="\r")
            return total

        total = asyncio.run(run_edge())
        print(f"\nedge-tts done: {total} files")
    except Exception as e:
        print(f"edge-tts failed: {e}")

print(f"\nReal : {len(list(REAL_DIR.glob('*.wav')))} files")
print(f"Fake : kokoro={len(list(kokoro_dir.glob('*.wav')))}  edge_tts={len(list(edgetts_dir.glob('*.wav')))}")

# =============================================================================
# CELL 5 — Build Dataset Manifest
# =============================================================================
import csv, random
from pathlib import Path

random.seed(42)

REAL_DIR = Path("/kaggle/working/real_audio")
FAKE_DIR = Path("/kaggle/working/fake_audio")
MANIFEST = Path("/kaggle/working/dataset.csv")

real_files = [(str(p), 0, "librispeech") for p in sorted(REAL_DIR.glob("*.wav"))]
fake_files = []
for subdir in sorted(FAKE_DIR.iterdir()):
    if subdir.is_dir():
        wavs = [(str(p), 1, subdir.name) for p in sorted(subdir.glob("*.wav"))]
        fake_files.extend(wavs)
        print(f"  {subdir.name:<20} {len(wavs)} fake clips")

# Balance real/fake
min_count  = min(len(real_files), len(fake_files))
real_files = random.sample(real_files, min_count)
fake_files = random.sample(fake_files, min_count)

all_files  = real_files + fake_files
random.shuffle(all_files)

n       = len(all_files)
n_val   = int(n * 0.15)
n_test  = int(n * 0.15)
val_set  = all_files[:n_val]
test_set = all_files[n_val:n_val + n_test]
train_set = all_files[n_val + n_test:]

with open(MANIFEST, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "source", "split"])
    for path, label, source in train_set:
        writer.writerow([path, label, source, "train"])
    for path, label, source in val_set:
        writer.writerow([path, label, source, "val"])
    for path, label, source in test_set:
        writer.writerow([path, label, source, "test"])

print(f"\nManifest saved → {MANIFEST}")
print(f"  Train : {len(train_set)}")
print(f"  Val   : {len(val_set)}")
print(f"  Test  : {len(test_set)}")
print(f"  Total : {n}")

# =============================================================================
# CELL 6 — Train RawNet2
# =============================================================================
import csv, random, sys, importlib.util, shutil
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Config ──
SEED        = 42
SAMPLE_RATE = 16000
SEGMENT_LEN = 64600
BATCH       = 24
EPOCHS      = 20
LR          = 1e-3
MANIFEST    = Path("/kaggle/working/dataset.csv")
CKPT        = Path("/kaggle/working/rawnet2_best.pth")   # saves directly to Output tab
LOG_PATH    = Path("/kaggle/working/rawnet2_log.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_CONFIG = {
    "nb_samp":      SEGMENT_LEN,
    "first_conv":   1024,
    "in_channels":  1,
    "filts":        [20, [20, 20], [20, 128], [128, 128]],
    "blocks":       [2, 4],
    "nb_fc_node":   1024,
    "gru_node":     1024,
    "nb_gru_layer": 3,
    "nb_classes":   2,
}

# ── Load RawNet2 model ──
vare_dir = "/kaggle/working/VARE"
matches  = list(Path(vare_dir).rglob("RawNet2Spoof.py"))
if not matches:
    raise FileNotFoundError(f"RawNet2Spoof.py not found under {vare_dir}")
spec = importlib.util.spec_from_file_location("RawNet2Spoof", str(matches[0]))
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
RawNet2 = mod.Model
print(f"RawNet2Spoof loaded from: {matches[0]}")

# ── Dataset ──
def read_manifest(split):
    files, labels = [], []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            if row["split"] == split and Path(row["path"]).exists():
                files.append(row["path"])
                labels.append(int(row["label"]))
    real = labels.count(0); fake = labels.count(1)
    print(f"  [{split}] {len(files)} samples | real={real} fake={fake}")
    return files, labels

class AudioDataset(Dataset):
    def __init__(self, files, labels):
        self.files  = files
        self.labels = labels

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            audio, sr = torchaudio.load(self.files[idx])
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            if sr != SAMPLE_RATE:
                audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
            if audio.shape[1] < SEGMENT_LEN:
                audio = torch.nn.functional.pad(audio, (0, SEGMENT_LEN - audio.shape[1]))
            else:
                start = random.randint(0, audio.shape[1] - SEGMENT_LEN)
                audio = audio[:, start:start + SEGMENT_LEN]
        except Exception:
            audio = torch.zeros(1, SEGMENT_LEN)
        return audio.squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def log(msg):
    print(msg)
    sys.stdout.flush()
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

# ── Build loaders ──
print("Loading manifest...")
train_f, train_l = read_manifest("train")
val_f,   val_l   = read_manifest("val")

counts  = [train_l.count(0), train_l.count(1)]
weights = [1.0 / counts[l] for l in train_l]
sampler = WeightedRandomSampler(weights, len(train_l), replacement=True)

train_loader = DataLoader(AudioDataset(train_f, train_l), batch_size=BATCH,
                          sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(AudioDataset(val_f, val_l),   batch_size=BATCH,
                          shuffle=False, num_workers=2, pin_memory=True)

# ── Build model ──
import copy
model = RawNet2(copy.deepcopy(MODEL_CONFIG)).to(device)
log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=4)

best_acc = 0.0

log(f"\nTraining RawNet2 for {EPOCHS} epochs | batch={BATCH} | device={device}")
log("=" * 60)

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    t_loss = t_correct = t_total = 0

    for i, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(audio)[1]
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss    += loss.item()
        t_correct += (logits.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)
        if i % 50 == 0:
            print(f"  Ep {epoch:02d} | {i:4d}/{len(train_loader)} | loss {loss.item():.4f}", end="\r")

    # ── Validate ──
    model.eval()
    v_correct = v_total = tp = fp = fn = 0
    with torch.no_grad():
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)
            logits = model(audio)[1]
            preds  = logits.argmax(1)
            v_correct += (preds == labels).sum().item()
            v_total   += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    val_acc   = v_correct / v_total
    recall    = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    scheduler.step(val_acc)

    log(f"Ep {epoch:02d}/{EPOCHS} | loss {t_loss/len(train_loader):.4f} | "
        f"train_acc {t_correct/t_total:.3f} | val_acc {val_acc:.3f} | "
        f"recall {recall:.3f} | prec {precision:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        # Save checkpoint — key is 'model_state' to match app.py
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "model_config": MODEL_CONFIG,
            "val_acc":      val_acc,
        }, CKPT)
        log(f"  ✓ Best saved → {CKPT}  (val_acc={val_acc:.3f})")

log(f"\nDone. Best val_acc: {best_acc:.3f}")
log(f"Checkpoint: {CKPT}")
print(f"\nCheckpoint size: {CKPT.stat().st_size / 1e6:.1f} MB")
print("\nDownload: Kaggle Output tab (right panel) → rawnet2_best.pth")
print("Then put it in: finetune/checkpoints/rawnet2_trained_best.pth")
