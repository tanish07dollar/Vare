# =============================================================================
# VARE — AASIST3 Fine-tuning Only
# Paste each CELL block into a separate Kaggle notebook cell.
# GPU required: T4 (15 GB VRAM)
# Total expected runtime: ~40-55 minutes
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
print("=" * 50)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Go to Settings → Accelerator → GPU T4 x1")

# =============================================================================
# CELL 2 — Install Dependencies
# =============================================================================
import subprocess

packages = ["kokoro>=0.9.2", "edge-tts", "nest_asyncio", "soundfile"]
for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.run(["pip", "install", "-q", pkg], check=True)
print("\nAll packages installed.")

# =============================================================================
# CELL 3 — Clone VARE from GitHub
# =============================================================================
from kaggle_secrets import UserSecretsClient
import subprocess, os, sys, shutil

GITHUB_USER = "tanish07dollar"
GITHUB_REPO = "Vare"

token    = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_url = f"https://{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
vare_dir = "/kaggle/working/VARE"

if os.path.exists(vare_dir):
    shutil.rmtree(vare_dir)

subprocess.run(["git", "clone", repo_url, vare_dir], check=True)
print("Repo cloned.")

# Add model paths
sys.path.insert(0, vare_dir)
sys.path.insert(0, f"{vare_dir}/models/aasist3")
print(f"VARE path added: {vare_dir}")

# =============================================================================
# CELL 4 — Download LibriSpeech + Generate Fake Audio
# =============================================================================
import asyncio, soundfile as sf
import torchaudio
from pathlib import Path

REAL_DIR = Path("/kaggle/working/real_audio")
FAKE_DIR = Path("/kaggle/working/fake_audio")
REAL_DIR.mkdir(exist_ok=True)
FAKE_DIR.mkdir(exist_ok=True)

# ── LibriSpeech real audio ──
print("Downloading LibriSpeech dev-clean (~337 MB)...")
dataset = torchaudio.datasets.LIBRISPEECH(
    root="/kaggle/working", url="dev-clean", download=True)
print(f"LibriSpeech: {len(dataset)} clips")

print("Exporting to wav...")
for i, (waveform, sr, _, _, _, _) in enumerate(dataset):
    out = REAL_DIR / f"real_{i:05d}.wav"
    if not out.exists():
        torchaudio.save(str(out), waveform, sr)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(dataset)}", end="\r")
print(f"\nReal audio ready: {len(list(REAL_DIR.glob('*.wav')))} files")

# ── Fake audio — Kokoro ──
N_FAKE = 500

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "To be or not to be, that is the question whether tis nobler in the mind.",
    "I wandered lonely as a cloud that floats on high over vales and hills.",
    "All that glitters is not gold, often have you heard that told.",
    "It was the best of times it was the worst of times it was the age of wisdom.",
    "Call me Ishmael. Some years ago I thought I would sail about a little.",
    "It is a truth universally acknowledged that a single man needs a fortune.",
] * 60

kokoro_dir = FAKE_DIR / "kokoro"
kokoro_dir.mkdir(exist_ok=True)
done_k = len(list(kokoro_dir.glob("*.wav")))

if done_k >= N_FAKE:
    print(f"Kokoro: {done_k} files already done, skipping.")
else:
    print(f"\nGenerating {N_FAKE} Kokoro clips...")
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")
        voices   = ["af_heart", "af_sky", "am_adam"]
        gen = done_k
        for i, text in enumerate(TEXTS[done_k:N_FAKE], start=done_k):
            try:
                for _, _, audio in pipeline(text, voice=voices[i % 3], speed=1.0):
                    sf.write(str(kokoro_dir / f"kokoro_{i:04d}.wav"), audio, 24000)
                    gen += 1; break
            except Exception:
                continue
            if (i + 1) % 50 == 0:
                print(f"  {gen}/{N_FAKE}", end="\r")
        print(f"\nKokoro done: {gen} files")
    except Exception as e:
        print(f"Kokoro failed: {e}")

# ── Fake audio — edge-tts ──
edgetts_dir = FAKE_DIR / "edge_tts"
edgetts_dir.mkdir(exist_ok=True)
done_e = len(list(edgetts_dir.glob("*.wav")))

if done_e >= N_FAKE:
    print(f"edge-tts: {done_e} files already done, skipping.")
else:
    print(f"\nGenerating {N_FAKE} edge-tts clips...")
    try:
        import nest_asyncio, edge_tts
        nest_asyncio.apply()
        VOICES = ["en-US-JennyNeural","en-US-GuyNeural","en-GB-SoniaNeural","en-AU-NatashaNeural"]

        async def gen_one(text, voice, path):
            try:
                tmp = str(path).replace(".wav", ".mp3")
                await edge_tts.Communicate(text, voice).save(tmp)
                w, sr = torchaudio.load(tmp)
                if w.shape[0] > 1: w = w.mean(0, keepdim=True)
                torchaudio.save(str(path), w, sr)
                os.remove(tmp); return True
            except: return False

        async def run():
            total = done_e
            for i in range(0, N_FAKE - done_e, 10):
                batch = TEXTS[done_e + i: done_e + i + 10]
                tasks = [gen_one(t, VOICES[(done_e+i+j) % 4],
                                 edgetts_dir / f"edgetts_{done_e+i+j:04d}.wav")
                         for j, t in enumerate(batch)]
                total += sum(await asyncio.gather(*tasks))
                print(f"  {total}/{N_FAKE}", end="\r")
            return total

        total = asyncio.run(run())
        print(f"\nedge-tts done: {total} files")
    except Exception as e:
        print(f"edge-tts failed: {e}")

print(f"\nReal: {len(list(REAL_DIR.glob('*.wav')))}  "
      f"Fake: kokoro={len(list(kokoro_dir.glob('*.wav')))}  "
      f"edge_tts={len(list(edgetts_dir.glob('*.wav')))}")

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
for d in sorted(FAKE_DIR.iterdir()):
    if d.is_dir():
        wavs = [(str(p), 1, d.name) for p in sorted(d.glob("*.wav"))]
        fake_files.extend(wavs)
        print(f"  {d.name:<20} {len(wavs)} clips")

min_count  = min(len(real_files), len(fake_files))
real_files = random.sample(real_files, min_count)
fake_files = random.sample(fake_files, min_count)

all_files = real_files + fake_files
random.shuffle(all_files)

n      = len(all_files)
n_val  = int(n * 0.15)
n_test = int(n * 0.15)

val_set   = all_files[:n_val]
test_set  = all_files[n_val:n_val + n_test]
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

print(f"\nManifest: train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

# =============================================================================
# CELL 6 — Fine-tune AASIST3
# =============================================================================
import csv, random, sys, copy
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
BATCH       = 6       # small batch — AASIST3 is VRAM-heavy
EPOCHS      = 15
LR          = 5e-5    # low LR for fine-tuning pretrained model
MANIFEST    = Path("/kaggle/working/dataset.csv")
CKPT        = Path("/kaggle/working/aasist3_finetuned.pth")
LOG_PATH    = Path("/kaggle/working/aasist3_log.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Load AASIST3 model code from VARE ──
import importlib.util, types

vare_dir    = "/kaggle/working/VARE"
model_dir   = Path(f"{vare_dir}/models/aasist3/model")

# Clear any stale 'model' entries from previous imports
for k in list(sys.modules):
    if k == "model" or k.startswith("model."):
        del sys.modules[k]

# Manually register the 'model' package by absolute path
# (sys.path tricks are unreliable in Kaggle for nested packages)
spec = importlib.util.spec_from_file_location(
    "model",
    str(model_dir / "__init__.py"),
    submodule_search_locations=[str(model_dir)]
)
model_pkg = importlib.util.module_from_spec(spec)
sys.modules["model"] = model_pkg
spec.loader.exec_module(model_pkg)

aasist3 = model_pkg.aasist3
print(f"AASIST3 model code loaded from: {model_dir}")

# ── Dataset ──
def read_manifest(split):
    files, labels = [], []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            if row["split"] == split and Path(row["path"]).exists():
                files.append(row["path"])
                labels.append(int(row["label"]))
    print(f"  [{split}] {len(files)} samples | real={labels.count(0)} fake={labels.count(1)}")
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
    print(msg); sys.stdout.flush()
    with open(LOG_PATH, "a") as f: f.write(msg + "\n")

# ── Build loaders ──
print("Loading manifest...")
train_f, train_l = read_manifest("train")
val_f,   val_l   = read_manifest("val")

counts  = [train_l.count(0), train_l.count(1)]
weights = [1.0 / counts[l] for l in train_l]
sampler = WeightedRandomSampler(weights, len(train_l), replacement=True)

train_loader = DataLoader(AudioDataset(train_f, train_l), batch_size=BATCH,
                          sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(AudioDataset(val_f, val_l), batch_size=BATCH,
                          shuffle=False, num_workers=2, pin_memory=True)

# ── Load pretrained AASIST3 ──
log("Loading pretrained AASIST3 from MTUCI/AASIST3...")
model = aasist3.from_pretrained("MTUCI/AASIST3").to(device)
torch.cuda.empty_cache()
log("Pretrained weights loaded.")

# ── Freeze backbone, unfreeze head only ──
for p in model.parameters():
    p.requires_grad = False

head_keywords = ["out_layer", "master", "inference_branch"]
unfrozen = 0
for name, p in model.named_parameters():
    if any(k in name.lower() for k in head_keywords):
        p.requires_grad = True
        unfrozen += 1

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
log(f"Unfrozen params: {unfrozen} tensors | {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0

log(f"\nFine-tuning AASIST3 | epochs={EPOCHS} | batch={BATCH} | lr={LR}")
log("=" * 60)

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    t_loss = t_correct = t_total = 0

    for i, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(audio)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss    += loss.item()
        t_correct += (logits.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)
        if i % 20 == 0:
            print(f"  Ep {epoch:02d} | {i:4d}/{len(train_loader)} | loss {loss.item():.4f}", end="\r")

    # ── Validate ──
    model.eval()
    v_correct = v_total = tp = fp = fn = 0
    with torch.no_grad():
        for audio, labels in val_loader:
            audio, labels = audio.to(device), labels.to(device)
            logits = model(audio)
            preds  = logits.argmax(1)
            v_correct += (preds == labels).sum().item()
            v_total   += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    val_acc   = v_correct / v_total
    recall    = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    scheduler.step()

    log(f"Ep {epoch:02d}/{EPOCHS} | loss {t_loss/len(train_loader):.4f} | "
        f"train_acc {t_correct/t_total:.3f} | val_acc {val_acc:.3f} | "
        f"recall {recall:.3f} | prec {precision:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_acc":     val_acc,
        }, CKPT)
        log(f"  ✓ Best saved → {CKPT}  (val_acc={val_acc:.3f})")

log(f"\nDone. Best val_acc: {best_acc:.3f}")
print(f"\nCheckpoint size: {CKPT.stat().st_size / 1e6:.1f} MB")
print("\nDownload: Kaggle Output tab → aasist3_finetuned.pth")
print("Then put it in: finetune/checkpoints/aasist3_finetuned.pth")
