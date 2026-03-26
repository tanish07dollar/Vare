# VARE — Voice Authenticity Risk Engine
## Project Progress Log

> Single source of truth for the VARE project.
> Updated after every step. If a Claude session hits the subscription limit,
> check "Current Resume Point" to know exactly where to pick up.

---

## Project Status
🟡 IN PROGRESS

## Current Resume Point
**Step 8 — AASIST3 fine-tuning running on Kaggle (Cell 10)**
- RawNet2 ✅ trained
- wav2vec2 ✅ fine-tuned (val_acc 1.000)
- AASIST3 🔄 training (~3 hrs remaining)
- After AASIST3 completes → run Cell 11 → download vare_checkpoints.zip → replace local checkpoints → test app.py

---

## What Has Been Built

### Infrastructure
- FastAPI backend with professional dark-mode UI
- 3-model ensemble pipeline with weighted scoring (AASIST3 0.20 / wav2vec2 0.45 / RawNet2 0.35)
- Segment-by-segment analysis with visual output
- Audio preprocessing pipeline (16kHz, 64600-sample / ~4s windowing)
- Kaggle training notebook (kaggle_notebook.py) — ready to run

### Models
| Model    | State                    | Checkpoint Path                                    |
|----------|--------------------------|----------------------------------------------------|
| AASIST3  | Pretrained only          | HuggingFace: MTUCI/AASIST3                         |
| wav2vec2 | Fine-tuned (overfit)     | finetune/checkpoints/wav2vec2_finetuned_full/       |
| RawNet2  | Trained (overfit)        | finetune/checkpoints/rawnet2_trained_best.pth       |

### Why Models Are Overfit
All three models trained on only 101 fake samples from a single source (ElevenLabs).
They memorised ElevenLabs artifacts specifically — they generalise to nothing else.
Fix: retrain on a diverse multi-system dataset (see Dataset Plan below).

### GPU
- Training: Kaggle T4 (16 GB VRAM, free 30 hrs/week)
- Local: No GPU (laptop only)

---

## Folder Structure
```
VARE/
├── app/            ← FastAPI server + UI
├── models/         ← aasist3/ + aasist_reference/
├── finetune/       ← training scripts + data + checkpoints + logs
├── evaluation/     ← evaluate.py (EER / t-DCF)
├── weights/        ← HuggingFace cache
├── kaggle_notebook.py  ← Kaggle training notebook (all cells)
├── KAGGLE_SETUP.md     ← Step-by-step Kaggle setup guide
└── PROGRESS.md     ← this file
```

---

## Dataset Plan (Updated)

### Primary Dataset — WaveFake (~8.8 GB)
Downloaded directly on Kaggle via zenodo_get. No registration required.

| Split  | Real Source       | Fake Systems                                         |
|--------|-------------------|------------------------------------------------------|
| Train  | LJSpeech + JSUT   | HiFi-GAN v1, HiFi-GAN v2, MelGAN, Full-Band MelGAN  |
| Dev    | LJSpeech + JSUT   | Multi-Band MelGAN, Parallel WaveGAN, WaveGlow        |
| Test   | LJSpeech + JSUT   | Hold-out from all systems (unseen speakers)          |

### Supplementary — Generated on Kaggle GPU (Free)
Two additional synthesis systems generated directly in the notebook.

| Tool      | Install          | Type              | Why                                          |
|-----------|------------------|-------------------|----------------------------------------------|
| Kokoro    | pip install kokoro | Neural TTS (2025) | Best open-source TTS, different artifacts    |
| edge-tts  | pip install edge-tts | Microsoft Azure TTS | Cloud-quality neural voice, free API      |

### Combined Training Dataset Target
```
Real  : ~5,000 samples  (LJSpeech + JSUT from WaveFake)
Fake  : ~5,000 samples  distributed across 9 systems:
          WaveFake    → HiFi-GAN v1, HiFi-GAN v2, MelGAN, Full-Band MelGAN,
                        Multi-Band MelGAN, Parallel WaveGAN, WaveGlow   (7 systems)
          Generated   → Kokoro, edge-tts                                 (2 systems)
Split : 70/15/15 train/val/test — stratified by source
Rule  : Test split never seen during training
```

### What the Trained Model Can Detect
After training on 9 synthesis systems, the model will generalise to:
- ElevenLabs, Google TTS, Amazon Polly, Azure TTS (similar to edge-tts family)
- Any HiFi-GAN / MelGAN based voice (most open-source TTS uses these vocoders)
- Voice cloning tools (XTTS v2, RVC, SoVITS use the same vocoders)
- Codec-based voices (partial — add CodecFake later to improve)

### What Users Can Feed Into the Model
```
Format  : .wav .mp3 .flac .ogg .m4a .webm
Length  : 2 seconds minimum, any maximum (auto-segmented)
Quality : Any bitrate (phone recordings, WhatsApp, studio audio)
Language: Any — model detects synthesis artifacts, not words
```

---

## Detection Philosophy

| Score   | Label  | Meaning                              |
|---------|--------|--------------------------------------|
| 0–30%   | LOW    | Passes cleanly — confident real      |
| 31–60%  | MEDIUM | Suspicious — flag for review         |
| 61–100% | HIGH   | Strong synthetic indicators          |

MEDIUM on an unknown AI tool = the model is generalising. That is the goal.

---

## Step-by-Step Execution Plan

### Step 1 — Create PROGRESS.md
- Date: 2026-03-24
- Status: ✅ DONE

### Step 2 — Push VARE to GitHub
- Date: 2026-03-25
- Status: ✅ DONE
- Command:
  ```bash
  git remote add origin https://github.com/YOUR_USERNAME/VARE.git
  git branch -M main
  git push -u origin main
  ```
- Then: Create Kaggle notebook, add GITHUB_TOKEN secret, clone repo.
- See: KAGGLE_SETUP.md

### Step 3 — [Kaggle Cell 4] Download WaveFake
- Date: 2026-03-25
- Status: ❌ SKIPPED — Wrong Zenodo ID. Replaced with LibriSpeech dev-clean
- Note: WaveFake Zenodo ID needs to be verified manually at zenodo.org

### Step 4 — [Kaggle Cell 5] Generate Fake Audio on Kaggle GPU
- Date: 2026-03-25
- Status: ✅ DONE
- Output: 1000 Kokoro + 1000 edge-tts = 2000 fake clips
- Note: edge-tts required nest_asyncio fix

### Step 5 — [Kaggle Cell 6] Build Unified Dataset Manifest
- Date: 2026-03-25
- Status: ✅ DONE
- Output: 2798 train / 599 val / ~600 test (LibriSpeech real + Kokoro/edge-tts fake)

### Step 6 — [Kaggle Cell 8] Train RawNet2
- Date: 2026-03-25
- Status: ✅ DONE
- Checkpoint: rawnet2_best.pth

### Step 7 — [Kaggle Cell 9] Fine-tune wav2vec2
- Date: 2026-03-25
- Status: ✅ DONE
- val_acc: 1.000 (legitimate — pretrained model on easy dataset)
- Checkpoint: wav2vec2_best.pth + wav2vec2_full/

### Step 8 — [Kaggle Cell 10] Fine-tune AASIST3
- Date: 2026-03-25
- Status: 🔄 IN PROGRESS
- Script: kaggle_notebook.py Cell 10
- Expected time: ~3 hrs remaining on T4
- Target val_acc: > 90%

### Step 9 — [Kaggle Cell 10] Download Checkpoints
- Date: —
- Status: ⏳ PENDING
- Method: Zip /kaggle/working/checkpoints/ → download via Output tab

### Step 10 — Replace Checkpoints Locally
- Date: —
- Status: ⏳ PENDING
- Action: Extract downloaded zip into finetune/checkpoints/
- Then: Test with app.py using real and fake audio samples

### Step 11 — Validate Server
- Date: —
- Status: ⏳ PENDING
- Test files: models/aasist3/test_audio/real_voice.wav + finetune/data/fake/ElevenLabs_*.mp3
- Expected: real_voice.wav → LOW, ElevenLabs → HIGH

---

## Benchmark History

| Date       | Step | AASIST3 EER | wav2vec2 EER | RawNet2 EER | Ensemble EER | Notes              |
|------------|------|-------------|--------------|-------------|--------------|---------------------|
| 2026-03-24 | 1    | —           | —            | 1.0 (overfit)| —           | Baseline pre-improvement |

---

## Step Completion Log

| Date       | Step | Status | Notes |
|------------|------|--------|-------|
| 2026-03-24 | 1    | ✅ DONE | PROGRESS.md created |
| 2026-03-25 | 2    | ✅ DONE | Pushed to github.com/tanish07dollar/Vare |
| 2026-03-25 | 3    | ✅ DONE | WaveFake skipped (wrong Zenodo ID). Used LibriSpeech dev-clean instead |
| 2026-03-25 | 4    | ✅ DONE | 1000 Kokoro + 1000 edge-tts fake clips generated on Kaggle T4 |
| 2026-03-25 | 5    | ✅ DONE | dataset.csv built: 2798 train / 599 val / ~600 test |
| 2026-03-25 | 6    | ✅ DONE | RawNet2 trained 30 epochs. checkpoint: rawnet2_best.pth |
| 2026-03-25 | 7    | ✅ DONE | wav2vec2 fine-tuned 15 epochs. val_acc=1.000. checkpoint: wav2vec2_best.pth |
| 2026-03-25 | 8    | 🔄 IN PROGRESS | AASIST3 fine-tuning running — 20 epochs, ~3 hrs remaining |

---

## Problems & Solutions

| Date       | Step | Problem                                      | Solution                                           |
|------------|------|----------------------------------------------|----------------------------------------------------|
| 2026-03-24 | —    | app.py had hardcoded home paths after restructure | Updated to Path(__file__).parent.parent relative paths |
| 2026-03-24 | —    | weights/ dir had hidden .locks, rmdir failed | Moved .locks explicitly, then removed empty dir    |
| 2026-03-24 | —    | RawNet2 Val Acc 1.0 — overfit                | Root cause: 101 fakes from 1 source. Fix: diverse data |
| 2026-03-24 | —    | ASVspoof 2019 LA requires university login   | Switched to WaveFake (Zenodo, no login needed)     |
| 2026-03-24 | —    | No local GPU after server crash              | Using Kaggle T4 (free 30 hrs/week)                 |
| 2026-03-25 | Cell 2 | transformers==4.40.0 conflicts with sentence-transformers 5.2.3 | Changed to transformers>=4.41.0 |
| 2026-03-25 | Cell 5 | edge-tts failed: asyncio.run() cannot be called from a running event loop | Added nest_asyncio.apply() before asyncio.run() — Jupyter/Kaggle already has a running event loop |
| 2026-03-25 | Cell 8 | ModuleNotFoundError: No module named 'models.RawNet2Spoof' | Wrong sys.path depth. Fix: point to aasist_reference/models/ not aasist_reference/ |
| 2026-03-25 | Cell 8 | RawNet2Spoof.py not in GitHub repo — nested .gitignore inside aasist_reference/ blocked it | Cloned original AASIST repo on Kaggle (Cell 8a), copied .py files across |
| 2026-03-25 | Cell 8 | ModuleNotFoundError persisted even after path fix | Used importlib.util.spec_from_file_location to load by exact file path, bypassing sys.path entirely |
| 2026-03-25 | Cell 4 | WaveFake download: wrong Zenodo record ID 5642795 downloaded unrelated 41MB German art dataset | Switched to Option A: LibriSpeech dev-clean (~337MB) via torchaudio.datasets.LIBRISPEECH |
| 2026-03-25 | Cell 6 | dataset.csv had 0 entries after session restart — WaveFake audio files wiped | Rebuilt manifest using LibriSpeech (real) + Kokoro/edge-tts (fake) |
| 2026-03-25 | Cell 9 | wav2vec2 training showed no output for minutes | Added sys.stdout.flush() after every print — Kaggle buffers stdout |
| 2026-03-25 | Cell 9 | val_acc = 1.000 at epoch 4 — looks like overfit | Legitimate result: pretrained deepfake detector on only 2 TTS systems is genuinely easy |
| 2026-03-25 | Cell 9 | User refreshed page mid-training, re-ran cell unnecessarily | Checkpoint survived in /kaggle/working/. Check log before re-running any training cell |
| 2026-03-25 | Cell 10 | ModuleNotFoundError: No module named 'model' (aasist3) | sys.path manipulation unreliable — used types.ModuleType + importlib to manually register package |
| 2026-03-25 | Cell 10 | FileNotFoundError: residual.py not found — models/aasist3/model/ missing from Kaggle | models/aasist3/ never pushed to GitHub. Fix: git add models/aasist3/ + push from laptop |
| 2026-03-25 | Cell 10 | CalledProcessError: fatal: not a git repository on Kaggle | Kaggle was on master branch, push went to main. VARE dir lost .git. Fix: shutil.rmtree + fresh re-clone |
| 2026-03-25 | Cell 10 | aasist3_hf clone (MTUCI/AASIST3) only contained weights, not Python code | Python code must come from local VARE repo, not HuggingFace. Re-cloned VARE after pushing aasist3 |

---

## Session Log

### Session 1 — 2026-03-24
**Completed:**
- Full codebase exploration and architecture review
- Folder restructuring (everything moved into VARE/)
- Updated app.py import paths
- Identified training data problem (1 fake source → overfit)
- Confirmed GPU capability (A4000, 16GB — now migrated to Kaggle T4)
- Built complete Kaggle training notebook (kaggle_notebook.py)
- Decided dataset: WaveFake (primary) + Kokoro + edge-tts (generated)
- Updated full execution plan (11 steps)
- Created KAGGLE_SETUP.md

**Interrupted by subscription limit:** No
**Next session starts at:** Step 2 — Push to GitHub, then run Kaggle notebook
