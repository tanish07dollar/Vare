# VARE — Voice Authenticity Risk Engine
## Project Progress Log

> Single source of truth for the VARE project.
> Updated after every step. If a Claude session hits the subscription limit,
> check "Current Resume Point" to know exactly where to pick up.

---

## Project Status
🟡 IN PROGRESS

## Current Resume Point
**Step 2 — Download LibriSpeech train-clean-100**
```bash
cd /home/Timble-Tanish/VARE
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz -P finetune/data/
```

---

## What Has Been Built

### Infrastructure
- FastAPI backend with professional dark-mode UI
- 3-model ensemble pipeline with weighted scoring (AASIST3 0.20 / wav2vec2 0.45 / RawNet2 0.35)
- Segment-by-segment analysis with visual output
- Audio preprocessing pipeline (16kHz, 64600-sample / ~4s windowing)

### Models
| Model    | State                    | Checkpoint Path                                    |
|----------|--------------------------|----------------------------------------------------|
| AASIST3  | Pretrained only          | HuggingFace: MTUCI/AASIST3                         |
| wav2vec2 | Fine-tuned (narrow)      | finetune/checkpoints/wav2vec2_finetuned_full/       |
| RawNet2  | Trained from scratch     | finetune/checkpoints/rawnet2_trained_best.pth       |

### Known Problems with Current Models
- All models trained on only 1 fake source (ElevenLabs)
- RawNet2 Val Acc = 1.0 → overfit, not generalising
- Will fail on any AI voice tool other than ElevenLabs

### GPU
- NVIDIA RTX A4000, 16 GB VRAM (~15.9 GB free)

---

## Folder Structure
```
VARE/
├── app/            ← FastAPI server + UI
├── models/         ← aasist3/ + aasist_reference/
├── finetune/       ← training scripts + data + checkpoints + logs
├── evaluation/     ← evaluate.py (EER / t-DCF)
├── weights/        ← HuggingFace cache
└── PROGRESS.md     ← this file
```

---

## Dataset Plan

### Download (Free)
| Dataset    | Year | Covers                                   | Source                                      |
|------------|------|------------------------------------------|---------------------------------------------|
| CodecFake  | 2024 | 15 codec models (VALL-E style, EnCodec)  | HuggingFace: rogertseng/CodecFake           |
| CodecFake+ | 2025 | 31 codec models + web-sourced eval       | HuggingFace: CodecFake/CodecFake_Plus_Dataset |
| ASVspoof 5 | 2024 | Modern deepfakes + adversarial attacks   | Zenodo: zenodo.org/records/14498691         |
| LibriSpeech train-clean-100 | — | 100hrs clean real speech    | openslr.org/resources/12                    |

### Generate Locally (Free)
| Tool    | Install            | Year | Why                                       |
|---------|--------------------|------|-------------------------------------------|
| XTTS v2 | pip install TTS    | 2023 | Voice cloning from 3s reference           |
| F5-TTS  | pip install f5-tts | 2024 | Near-ElevenLabs quality                   |
| Kokoro  | pip install kokoro | 2025 | Best open-source TTS available right now  |

### Target
```
Real : ~3,000 samples (LibriSpeech)
Fake : ~2,000+ samples across 6+ synthesis systems
Split: 70/15/15 train/val/test — stratified by source
Rule : Test split never seen during training
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

## Step-by-Step Execution Log

### Step 1 — Create PROGRESS.md
- Date: 2026-03-24
- Status: ✅ DONE

### Step 2 — Download LibriSpeech train-clean-100
- Date: —
- Status: ⏳ PENDING
- Command: `wget https://www.openslr.org/resources/12/train-clean-100.tar.gz -P finetune/data/`
- Expected output: ~6GB tar.gz in finetune/data/
- Result: —

### Step 3 — Download CodecFake (HuggingFace)
- Date: —
- Status: ⏳ PENDING
- Command: See session log when reached
- Result: —

### Step 4 — Generate fake audio locally
- Date: —
- Status: ⏳ PENDING
- Tools: XTTS v2 / F5-TTS / Kokoro
- Result: —

### Step 5 — Retrain RawNet2
- Date: —
- Status: ⏳ PENDING
- Script: finetune/train_rawnet2.py
- Result: —

### Step 6 — Re-finetune wav2vec2
- Date: —
- Status: ⏳ PENDING
- Script: finetune/finetune_wav2vec2.py
- Result: —

### Step 7 — Fine-tune AASIST3
- Date: —
- Status: ⏳ PENDING
- Script: finetune/finetune_aasist3.py
- Result: —

### Step 8 — Run Evaluation (EER)
- Date: —
- Status: ⏳ PENDING
- Script: evaluation/evaluate.py
- Result: —

### Step 9 — Build Stage 2 Spectral Classifier
- Date: —
- Status: ⏳ PENDING
- Result: —

### Step 10 — Validate Server with Test Audio
- Date: —
- Status: ⏳ PENDING
- Test files: models/aasist3/test_audio/real_voice.wav + finetune/data/fake/ElevenLabs_*.mp3
- Result: —

---

## Benchmark History

| Date       | Step | AASIST3 EER | wav2vec2 EER | RawNet2 EER | Ensemble EER | Notes              |
|------------|------|-------------|--------------|-------------|--------------|---------------------|
| 2026-03-24 | 1    | —           | —            | 1.0 (overfit)| —           | Baseline pre-improvement |

---

## Problems & Solutions

| Date       | Step | Problem                                      | Solution                                           |
|------------|------|----------------------------------------------|----------------------------------------------------|
| 2026-03-24 | —    | app.py had hardcoded home paths after restructure | Updated to Path(__file__).parent.parent relative paths |
| 2026-03-24 | —    | weights/ dir had hidden .locks, rmdir failed | Moved .locks explicitly, then removed empty dir    |
| 2026-03-24 | —    | RawNet2 Val Acc 1.0 — overfit                | Root cause: 101 fakes from 1 source. Fix: diverse data in Steps 2-4 |

---

## Session Log

### Session 1 — 2026-03-24
**Completed:**
- Full codebase exploration and architecture review
- Folder restructuring (everything moved into VARE/)
- Updated app.py import paths
- Identified training data problem (1 fake source)
- Confirmed GPU capability (A4000, 16GB — can fine-tune AASIST3)
- Updated dataset plan (CodecFake+, ASVspoof5 replacing old datasets)
- Finalised 10-step manual execution plan
- Created PROGRESS.md

**Interrupted by subscription limit:** No
**Next session starts at:** Step 2 — Download LibriSpeech
