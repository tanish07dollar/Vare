# =============================================================================
# VARE — Kaggle Training Notebook
# Paste each CELL block into a separate Kaggle notebook cell.
# GPU required: T4 (15 GB VRAM) | Session limit: 12 hrs
# Total expected runtime: ~10-11 hrs
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
print(f"Working dir     : /kaggle/working/")
print("=" * 50)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Go to Settings → Accelerator → GPU T4 x1")

# =============================================================================
# CELL 2 — Install Dependencies
# =============================================================================
# Runtime: ~3-4 minutes

import subprocess

packages = [
    "transformers>=4.41.0",
    "accelerate",
    "zenodo_get",        # for downloading WaveFake from Zenodo
    "kokoro>=0.9.2",     # neural TTS for fake generation
    "edge-tts",          # Microsoft Azure TTS for fake generation
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
# SETUP REQUIRED before running:
#   1. Push your local VARE folder to a GitHub repo (private is fine)
#   2. Kaggle → Add-ons → Secrets → Add secret:
#        Name:  GITHUB_TOKEN
#        Value: your GitHub personal access token (classic, repo scope)
#        Toggle "Notebook access" ON
#
# Get a token at: github.com → Settings → Developer settings →
#                 Personal access tokens → Tokens (classic) → Generate new token
#                 Scope: tick "repo" only

from kaggle_secrets import UserSecretsClient
import subprocess, os

GITHUB_USER = "YOUR_GITHUB_USERNAME"   # <-- CHANGE THIS
GITHUB_REPO = "VARE"                   # <-- change if your repo name is different

token    = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_url = f"https://{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
vare_dir = "/kaggle/working/VARE"

if not os.path.exists(vare_dir):
    subprocess.run(["git", "clone", repo_url, vare_dir], check=True)
    print("Repo cloned.")
else:
    subprocess.run(["git", "-C", vare_dir, "pull"], check=True)
    print("Repo updated.")

sys.path.insert(0, vare_dir)
sys.path.insert(0, f"{vare_dir}/models/aasist_reference")
sys.path.insert(0, f"{vare_dir}/models/aasist3")
print(f"VARE path added: {vare_dir}")

# =============================================================================
# CELL 4 — Download WaveFake (Zenodo, no login needed)
# =============================================================================
# WaveFake: 8.8 GB, 6 GAN vocoders, LJSpeech + JSUT real speech
# Zenodo DOI: 10.5281/zenodo.5642795
# Runtime: ~15-20 minutes on Kaggle

import subprocess
from pathlib import Path

WAVEFAKE_DIR = Path("/kaggle/working/wavefake")
WAVEFAKE_DIR.mkdir(exist_ok=True)

# Check if already downloaded (resumable)
existing = list(WAVEFAKE_DIR.glob("*.zip")) + list(WAVEFAKE_DIR.glob("*.tar.gz"))
if not existing:
    print("Downloading WaveFake from Zenodo (~8.8 GB)...")
    result = subprocess.run(
        ["zenodo_get", "5642795", "-o", str(WAVEFAKE_DIR)],
        capture_output=True, text=True
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr[-1000:])
else:
    print(f"Files already present: {[f.name for f in existing]}")

# Extract all zip/tar files
import zipfile, tarfile

for archive in sorted(WAVEFAKE_DIR.glob("*.zip")):
    dest = WAVEFAKE_DIR / archive.stem
    if not dest.exists():
        print(f"Extracting {archive.name}...")
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(WAVEFAKE_DIR)
    else:
        print(f"Already extracted: {archive.name}")

for archive in sorted(WAVEFAKE_DIR.glob("*.tar.gz")):
    dest = WAVEFAKE_DIR / archive.name.replace(".tar.gz", "")
    if not dest.exists():
        print(f"Extracting {archive.name}...")
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(WAVEFAKE_DIR)

# Verify structure
print("\n--- WaveFake directory contents ---")
for item in sorted(WAVEFAKE_DIR.iterdir()):
    if item.is_dir():
        wav_count = len(list(item.rglob("*.wav")))
        print(f"  {item.name:<40} {wav_count:>6} wav files")

# =============================================================================
# CELL 5 — Generate Additional Fake Audio (Kokoro + edge-tts)
# =============================================================================
# Uses LJSpeech transcripts from WaveFake as text input.
# Generates ~2000 fake clips across 2 synthesis systems.
# Runtime: ~30-45 minutes on T4 GPU

import asyncio, soundfile as sf, numpy as np
from pathlib import Path

GEN_DIR    = Path("/kaggle/working/generated_fakes")
N_GENERATE = 1000   # clips per TTS system (adjust if running out of time)

# --- Load LJSpeech transcripts (included in WaveFake download) ---
lj_meta = WAVEFAKE_DIR / "LJSpeech-1.1" / "metadata.csv"
if not lj_meta.exists():
    # Search for it
    candidates = list(WAVEFAKE_DIR.rglob("metadata.csv"))
    lj_meta = candidates[0] if candidates else None

transcripts = []
if lj_meta and lj_meta.exists():
    with open(lj_meta, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2 and len(parts[1]) > 10:
                transcripts.append(parts[1].strip())
    print(f"Loaded {len(transcripts)} LJSpeech transcripts")
else:
    # Fallback: use built-in sentences
    transcripts = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "Peter Piper picked a peck of pickled peppers.",
        "To be or not to be, that is the question.",
    ] * 300
    print(f"Using {len(transcripts)} fallback transcripts")

transcripts = transcripts[:N_GENERATE]


# ---- Part A: Kokoro ----
print("\n--- Generating with Kokoro ---")
kokoro_dir = GEN_DIR / "kokoro"
kokoro_dir.mkdir(parents=True, exist_ok=True)

already_done = len(list(kokoro_dir.glob("*.wav")))
if already_done >= N_GENERATE:
    print(f"Kokoro: {already_done} files already generated, skipping.")
else:
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")   # American English

        voices = ["af_heart", "af_sky", "am_adam"]  # 3 built-in voices for variety
        generated = already_done

        for i, text in enumerate(transcripts[already_done:], start=already_done):
            voice = voices[i % len(voices)]
            try:
                for _, _, audio in pipeline(text, voice=voice, speed=1.0):
                    out_path = kokoro_dir / f"kokoro_{i:04d}.wav"
                    sf.write(str(out_path), audio, 24000)
                    generated += 1
                    break  # one segment per transcript
            except Exception as e:
                continue

            if (i + 1) % 100 == 0:
                print(f"  Kokoro: {generated}/{N_GENERATE} generated", end="\r")

        print(f"\nKokoro done: {generated} files")
    except Exception as e:
        print(f"Kokoro failed: {e} — skipping")


# ---- Part B: edge-tts ----
print("\n--- Generating with edge-tts ---")
edgetts_dir = GEN_DIR / "edge_tts"
edgetts_dir.mkdir(parents=True, exist_ok=True)

already_done_e = len(list(edgetts_dir.glob("*.wav")))
if already_done_e >= N_GENERATE:
    print(f"edge-tts: {already_done_e} files already generated, skipping.")
else:
    try:
        import nest_asyncio
        nest_asyncio.apply()   # required in Kaggle/Jupyter — already-running event loop fix

        import edge_tts, asyncio, torchaudio

        edge_voices = [
            "en-US-JennyNeural",
            "en-US-GuyNeural",
            "en-GB-SoniaNeural",
            "en-AU-NatashaNeural",
        ]

        async def generate_edge(text, voice, out_path):
            try:
                communicate = edge_tts.Communicate(text, voice)
                tmp = str(out_path).replace(".wav", ".mp3")
                await communicate.save(tmp)
                waveform, sr = torchaudio.load(tmp)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)
                torchaudio.save(str(out_path), waveform, sr)
                os.remove(tmp)
                return True
            except Exception:
                return False

        async def run_all_edge():
            total = already_done_e
            for i in range(already_done_e, len(transcripts[:N_GENERATE]), 10):
                batch   = transcripts[i:i + 10]
                tasks   = [
                    generate_edge(
                        text,
                        edge_voices[(i + j) % len(edge_voices)],
                        edgetts_dir / f"edgetts_{i+j:04d}.wav"
                    )
                    for j, text in enumerate(batch)
                    if not (edgetts_dir / f"edgetts_{i+j:04d}.wav").exists()
                ]
                results = await asyncio.gather(*tasks)
                total  += sum(results)
                print(f"  edge-tts: {total}/{N_GENERATE}", end="\r")
            return total

        total_gen = asyncio.run(run_all_edge())
        print(f"\nedge-tts done: {total_gen} files")
    except Exception as e:
        print(f"edge-tts failed: {e} — skipping")

# Summary
print("\n--- Generated fake audio summary ---")
for d in sorted(GEN_DIR.iterdir()):
    if d.is_dir():
        count = len(list(d.glob("*.wav")))
        print(f"  {d.name:<20} {count} files")

# =============================================================================
# CELL 6 — Build Unified Dataset Manifest
# =============================================================================
# Scans WaveFake + generated fakes, assigns labels, creates balanced split.
# Output: /kaggle/working/dataset.csv

import csv, random
from pathlib import Path

random.seed(42)

WAVEFAKE_DIR = Path("/kaggle/working/wavefake")
GEN_DIR      = Path("/kaggle/working/generated_fakes")
MANIFEST     = Path("/kaggle/working/dataset.csv")

MAX_PER_SOURCE = 700   # cap samples per synthesis system (prevents any one system dominating)
MAX_REAL       = 5000  # total real samples

# ---- Collect REAL audio from WaveFake ----
real_sources = [
    ("LJSpeech-1.1", WAVEFAKE_DIR / "LJSpeech-1.1"),
    ("jsut",         WAVEFAKE_DIR / "jsut_basic5000"),
]

real_files = []
for source_name, source_dir in real_sources:
    if source_dir.exists():
        wavs = list(source_dir.rglob("*.wav"))
        random.shuffle(wavs)
        real_files.extend([(str(p), 0, source_name) for p in wavs])

random.shuffle(real_files)
real_files = real_files[:MAX_REAL]
print(f"Real samples collected: {len(real_files)}")

# ---- Collect FAKE audio from WaveFake ----
# Folder names follow pattern: ljspeech_<vocoder> or jsut_<vocoder>
fake_systems_wf = [
    d for d in WAVEFAKE_DIR.iterdir()
    if d.is_dir() and ("ljspeech_" in d.name or "jsut_" in d.name)
]

fake_files = []
for system_dir in sorted(fake_systems_wf):
    wavs = list(system_dir.rglob("*.wav"))
    random.shuffle(wavs)
    selected = wavs[:MAX_PER_SOURCE]
    fake_files.extend([(str(p), 1, system_dir.name) for p in selected])
    print(f"  {system_dir.name:<40} {len(selected)} samples")

# ---- Collect FAKE audio from generated sources ----
if GEN_DIR.exists():
    for gen_subdir in sorted(GEN_DIR.iterdir()):
        if gen_subdir.is_dir():
            wavs = list(gen_subdir.glob("*.wav"))
            random.shuffle(wavs)
            selected = wavs[:MAX_PER_SOURCE]
            fake_files.extend([(str(p), 1, gen_subdir.name) for p in selected])
            print(f"  {gen_subdir.name:<40} {len(selected)} samples")

print(f"\nFake samples collected: {len(fake_files)}")

# ---- Balance and split ----
# Balance: keep equal real/fake counts
min_count = min(len(real_files), len(fake_files))
real_files = real_files[:min_count]
fake_files = fake_files[:min_count]

all_files = real_files + fake_files
random.shuffle(all_files)

n      = len(all_files)
n_test = int(n * 0.15)
n_val  = int(n * 0.15)

test_set  = all_files[:n_test]
val_set   = all_files[n_test:n_test + n_val]
train_set = all_files[n_test + n_val:]

print(f"\nDataset split:")
print(f"  Train : {len(train_set)} samples")
print(f"  Val   : {len(val_set)}   samples")
print(f"  Test  : {len(test_set)}  samples")
print(f"  Total : {n} samples")

# ---- Write CSV manifest ----
with open(MANIFEST, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "source", "split"])
    for path, label, source in train_set:
        writer.writerow([path, label, source, "train"])
    for path, label, source in val_set:
        writer.writerow([path, label, source, "val"])
    for path, label, source in test_set:
        writer.writerow([path, label, source, "test"])

print(f"\nManifest saved: {MANIFEST}")
print(f"Total entries : {n}")

# =============================================================================
# CELL 7 — Shared Utilities  (run before any training cell)
# =============================================================================
import csv, random
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
MANIFEST    = Path("/kaggle/working/dataset.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def read_manifest(split):
    """Read dataset.csv and return (files, labels) for a given split."""
    files, labels = [], []
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split and Path(row["path"]).exists():
                files.append(row["path"])
                labels.append(int(row["label"]))
    real  = labels.count(0)
    fake  = labels.count(1)
    print(f"  [{split}] Total: {len(files)} | Real: {real} | Fake: {fake}")
    return files, labels


def load_audio(path):
    """Load audio file → fixed-length float32 tensor at 16kHz."""
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


class AudioDataset(Dataset):
    def __init__(self, files, labels):
        self.files  = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_audio(self.files[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def make_loaders(train_files, train_labels, val_files, val_labels, batch_size, num_workers=2):
    train_ds = AudioDataset(train_files, train_labels)
    val_ds   = AudioDataset(val_files,   val_labels)

    # Weighted sampler — handles any class imbalance automatically
    counts  = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def evaluate(model, loader, forward_fn):
    """Generic validation loop. Returns (acc, recall, precision)."""
    model.eval()
    correct = total = tp = fp = tn = fn = 0
    with torch.no_grad():
        for audio, labels in loader:
            audio, labels = audio.to(device), labels.to(device)
            logits  = forward_fn(model, audio)
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    acc       = correct / total
    recall    = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    return acc, recall, precision


def log(msg, path):
    print(msg)
    with open(path, "a") as f:
        f.write(msg + "\n")


print("Utilities ready.")

# =============================================================================
# CELL 8 — Train RawNet2
# =============================================================================
# Expected runtime: ~3-4 hours on T4
# Expected val_acc: > 90%

import sys, importlib.util
from pathlib import Path

vare_dir = "/kaggle/working/VARE"

# Locate RawNet2Spoof.py anywhere in the repo (robust to path differences)
matches = list(Path(vare_dir).rglob("RawNet2Spoof.py"))
if not matches:
    raise FileNotFoundError(f"RawNet2Spoof.py not found under {vare_dir}")
rawnet2_path = str(matches[0])
print(f"Loading RawNet2 from: {rawnet2_path}")

spec   = importlib.util.spec_from_file_location("RawNet2Spoof", rawnet2_path)
mod    = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
RawNet2 = mod.Model
print("RawNet2 loaded.")

LOG   = OUTPUT_DIR / "rawnet2_log.txt"
CKPT  = OUTPUT_DIR / "rawnet2_best.pth"

CONFIG = {
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

BATCH  = 24
EPOCHS = 30
LR     = 1e-3

def L(msg): log(msg, LOG)

L("=== RawNet2 Training ===")
print("Loading dataset...")
train_f, train_l = read_manifest("train")
val_f,   val_l   = read_manifest("val")
train_loader, val_loader = make_loaders(train_f, train_l, val_f, val_l, BATCH, num_workers=4)

model     = RawNet2(CONFIG).to(device)
L(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

best_acc = 0.0

def rn_forward(m, x): return m(x)[1]

L(f"Training for {EPOCHS} epochs | batch={BATCH} | device={device}")
L("=" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = t_correct = t_total = 0

    for i, (audio, labels) in enumerate(train_loader):
        audio, labels = audio.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = rn_forward(model, audio)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss    += loss.item()
        t_correct += (logits.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)
        if i % 100 == 0:
            print(f"  Ep {epoch:02d} | {i:4d}/{len(train_loader)} | loss {loss.item():.4f}", end="\r")

    val_acc, val_recall, val_prec = evaluate(model, val_loader, rn_forward)
    scheduler.step(val_acc)

    msg = (f"Ep {epoch:02d}/{EPOCHS} | "
           f"loss {t_loss/len(train_loader):.4f} | "
           f"train_acc {t_correct/t_total:.3f} | "
           f"val_acc {val_acc:.3f} | recall {val_recall:.3f} | prec {val_prec:.3f}")
    L(msg)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"epoch": epoch, "state": model.state_dict(),
                    "config": CONFIG, "val_acc": val_acc}, CKPT)
        L(f"  ✓ Best saved  val_acc={val_acc:.3f}")

L(f"\nRawNet2 done. Best val_acc: {best_acc:.3f} | Checkpoint: {CKPT}")

# =============================================================================
# CELL 9 — Fine-tune wav2vec2
# =============================================================================
# Expected runtime: ~4-5 hours on T4
# Expected val_acc: > 95%

import gc
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

LOG_W  = OUTPUT_DIR / "wav2vec2_log.txt"
CKPT_W = OUTPUT_DIR / "wav2vec2_best.pth"
SAVE_W = OUTPUT_DIR / "wav2vec2_full"

MODEL_HF   = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
BATCH_W    = 4     # large model — small batch needed on T4
EPOCHS_W   = 15
LR_W       = 5e-5

def LW(msg): log(msg, LOG_W)

LW("=== wav2vec2 Fine-tuning ===")

# Dataset with feature extractor
class W2VDataset(Dataset):
    def __init__(self, files, labels, extractor):
        self.files, self.labels, self.ext = files, labels, extractor

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        audio  = load_audio(self.files[idx]).numpy()
        inputs = self.ext(audio, sampling_rate=SAMPLE_RATE,
                          return_tensors="pt", padding=True)
        return inputs["input_values"].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)


LW(f"Loading model: {MODEL_HF}")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_HF)
w2v       = AutoModelForAudioClassification.from_pretrained(MODEL_HF).to(device)

# Freeze CNN + bottom 18/24 transformer layers
for p in w2v.wav2vec2.feature_extractor.parameters():
    p.requires_grad = False
for i, layer in enumerate(w2v.wav2vec2.encoder.layers):
    if i < 18:
        for p in layer.parameters():
            p.requires_grad = False

trainable = sum(p.numel() for p in w2v.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in w2v.parameters())
LW(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

print("Loading dataset...")
train_f_w, train_l_w = read_manifest("train")
val_f_w,   val_l_w   = read_manifest("val")

train_ds_w = W2VDataset(train_f_w, train_l_w, extractor)
val_ds_w   = W2VDataset(val_f_w,   val_l_w,   extractor)

counts_w  = [train_l_w.count(0), train_l_w.count(1)]
weights_w = [1.0 / counts_w[l] for l in train_l_w]
sampler_w = WeightedRandomSampler(weights_w, len(train_l_w), replacement=True)

train_loader_w = DataLoader(train_ds_w, batch_size=BATCH_W, sampler=sampler_w,
                            num_workers=2, pin_memory=True, drop_last=True)
val_loader_w   = DataLoader(val_ds_w,   batch_size=BATCH_W, shuffle=False,
                            num_workers=2, pin_memory=True)

optimizer_w = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, w2v.parameters()), lr=LR_W, weight_decay=1e-4)
scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_w, T_max=EPOCHS_W)
criterion   = torch.nn.CrossEntropyLoss()
best_acc_w  = 0.0

def w2v_fwd(m, x): return m(x).logits

LW(f"Fine-tuning for {EPOCHS_W} epochs | batch={BATCH_W}")
LW("=" * 60)

for epoch in range(1, EPOCHS_W + 1):
    w2v.train()
    t_loss = t_correct = t_total = 0

    for i, (inputs, labels) in enumerate(train_loader_w):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_w.zero_grad()
        logits = w2v_fwd(w2v, inputs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(w2v.parameters(), 1.0)
        optimizer_w.step()
        t_loss    += loss.item()
        t_correct += (logits.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)
        if i % 50 == 0:
            print(f"  Ep {epoch:02d} | {i:4d}/{len(train_loader_w)} | loss {loss.item():.4f}", end="\r")

    scheduler_w.step()
    val_acc, val_recall, val_prec = evaluate(w2v, val_loader_w, w2v_fwd)

    msg = (f"Ep {epoch:02d}/{EPOCHS_W} | "
           f"loss {t_loss/len(train_loader_w):.4f} | "
           f"train_acc {t_correct/t_total:.3f} | "
           f"val_acc {val_acc:.3f} | recall {val_recall:.3f} | prec {val_prec:.3f}")
    LW(msg)

    if val_acc > best_acc_w:
        best_acc_w = val_acc
        torch.save(w2v.state_dict(), CKPT_W)
        LW(f"  ✓ Best saved  val_acc={val_acc:.3f}")

# Save full model for app.py
w2v.save_pretrained(str(SAVE_W))
extractor.save_pretrained(str(SAVE_W))
LW(f"\nwav2vec2 done. Best val_acc: {best_acc_w:.3f}")

del w2v, train_ds_w, val_ds_w
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# CELL 10 — Fine-tune AASIST3
# =============================================================================
# Expected runtime: ~2-3 hours on T4
# Expected val_acc: > 95%

import sys, importlib
from pathlib import Path

vare_dir = "/kaggle/working/VARE"

# Clear any stale 'model' module cached from previous cells
stale = [k for k in sys.modules if k == "model" or k.startswith("model.")]
for k in stale:
    del sys.modules[k]
print(f"Cleared {len(stale)} stale module(s): {stale}")

aasist3_dir = str(Path(vare_dir) / "models" / "aasist3")
if aasist3_dir not in sys.path:
    sys.path.insert(0, aasist3_dir)

from model import aasist3
print("aasist3 loaded successfully.")

LOG_A  = OUTPUT_DIR / "aasist3_log.txt"
CKPT_A = OUTPUT_DIR / "aasist3_best.pth"

BATCH_A  = 8
EPOCHS_A = 20
LR_A     = 1e-4

def LA(msg): log(msg, LOG_A)

LA("=== AASIST3 Fine-tuning ===")

print("Loading dataset...")
train_f_a, train_l_a = read_manifest("train")
val_f_a,   val_l_a   = read_manifest("val")
train_loader_a, val_loader_a = make_loaders(
    train_f_a, train_l_a, val_f_a, val_l_a, BATCH_A, num_workers=4)

LA("Loading pretrained AASIST3 from MTUCI/AASIST3...")
a3 = aasist3.from_pretrained("MTUCI/AASIST3").to(device)
torch.cuda.empty_cache()

# Freeze all → unfreeze classifier head
for p in a3.parameters():
    p.requires_grad = False

head_keywords = ["out_layer", "master", "fc", "classifier", "head", "branch", "linear"]
unfrozen = 0
for name, p in a3.named_parameters():
    if any(k in name.lower() for k in head_keywords):
        p.requires_grad = True
        unfrozen += 1

if unfrozen == 0:
    # Fallback: unfreeze last 30% of parameters
    all_params = list(a3.parameters())
    for p in all_params[int(len(all_params) * 0.7):]:
        p.requires_grad = True
    LA("Fallback: unfroze last 30% of parameters")

trainable = sum(p.numel() for p in a3.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in a3.parameters())
LA(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

optimizer_a = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, a3.parameters()), lr=LR_A, weight_decay=1e-4)
scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=EPOCHS_A)
criterion   = torch.nn.CrossEntropyLoss()
best_acc_a  = 0.0

def a3_fwd(m, x): return m(x)

LA(f"Fine-tuning for {EPOCHS_A} epochs | batch={BATCH_A}")
LA("=" * 60)

for epoch in range(1, EPOCHS_A + 1):
    a3.train()
    t_loss = t_correct = t_total = 0

    for i, (audio, labels) in enumerate(train_loader_a):
        audio, labels = audio.to(device), labels.to(device)
        optimizer_a.zero_grad()
        logits = a3_fwd(a3, audio)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a3.parameters(), 1.0)
        optimizer_a.step()
        t_loss    += loss.item()
        t_correct += (logits.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)
        if i % 100 == 0:
            print(f"  Ep {epoch:02d} | {i:4d}/{len(train_loader_a)} | loss {loss.item():.4f}", end="\r")

    scheduler_a.step()
    val_acc, val_recall, val_prec = evaluate(a3, val_loader_a, a3_fwd)

    msg = (f"Ep {epoch:02d}/{EPOCHS_A} | "
           f"loss {t_loss/len(train_loader_a):.4f} | "
           f"train_acc {t_correct/t_total:.3f} | "
           f"val_acc {val_acc:.3f} | recall {val_recall:.3f} | prec {val_prec:.3f}")
    LA(msg)

    if val_acc > best_acc_a:
        best_acc_a = val_acc
        torch.save(a3.state_dict(), CKPT_A)
        LA(f"  ✓ Best saved  val_acc={val_acc:.3f}")

LA(f"\nAASIST3 done. Best val_acc: {best_acc_a:.3f}")

# =============================================================================
# CELL 11 — Final Summary & Download
# =============================================================================
import os, shutil
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/checkpoints")

print("\n" + "=" * 60)
print("VARE TRAINING COMPLETE")
print("=" * 60)

print("\n--- Checkpoint files ---")
for f in sorted(OUTPUT_DIR.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1e6
        print(f"  {str(f.relative_to(OUTPUT_DIR)):<45} {size_mb:>8.1f} MB")

print("\n--- Results ---")
try: print(f"  RawNet2  best val_acc : {best_acc:.3f}")
except: pass
try: print(f"  wav2vec2 best val_acc : {best_acc_w:.3f}")
except: pass
try: print(f"  AASIST3  best val_acc : {best_acc_a:.3f}")
except: pass

# Zip everything for download
zip_path = "/kaggle/working/vare_checkpoints"
shutil.make_archive(zip_path, "zip", OUTPUT_DIR)
zip_size = os.path.getsize(zip_path + ".zip") / 1e6
print(f"\nZipped: {zip_path}.zip  ({zip_size:.0f} MB)")
print("\nDownload from: Kaggle Output tab (right panel) → vare_checkpoints.zip")
print("\nAfter download, extract into: finetune/checkpoints/")
print("Then restart app.py — it will use the new models automatically.")
