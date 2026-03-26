# VARE — Voice Authenticity Risk Engine
## Project Progress Log

> Single source of truth for the VARE project.
> Updated after every step. If a Claude session hits the subscription limit,
> check "Current Resume Point" to know exactly where to pick up.

---

## Project Status
🟢 COMPLETE — All 3 models trained and running locally

## Current Resume Point
**Server is running. All 3 models loaded. Ready for deployment.**
- RawNet2   ✅ fine-tuned  (val_acc 1.000) — `finetune/checkpoints/rawnet2_trained_best.pth`
- wav2vec2  ✅ pretrained  (HuggingFace: Gustking/wav2vec2-large-xlsr-deepfake-audio-classification)
- AASIST3  ✅ fine-tuned  (val_acc 0.820) — `finetune/checkpoints/aasist3_finetuned.pth`

**Next steps (optional improvements):**
- Deploy server to cloud (Railway / Render / EC2)
- Add authentication to `/analyze` endpoint
- Improve generalisation: retrain on WaveFake (7 GAN systems) when time allows

---

## What Has Been Built

### Infrastructure
- FastAPI backend with professional dark-mode UI
- 3-model ensemble pipeline with weighted scoring (AASIST3 0.20 / wav2vec2 0.45 / RawNet2 0.35)
- Segment-by-segment analysis with visual output
- Audio preprocessing pipeline (16kHz, 64600-sample / ~4s windowing)
- Local Python environment (`vare_env/`) with all dependencies
- API integration reference file (`vare_api.http`) for colleague handoff

### Models
| Model    | State                        | val_acc | Checkpoint Path                                                        |
|----------|------------------------------|---------|------------------------------------------------------------------------|
| AASIST3  | Fine-tuned (Kaggle T4)       | 0.820   | finetune/checkpoints/aasist3_finetuned.pth                             |
| wav2vec2 | Pretrained (HuggingFace)     | —       | Gustking/wav2vec2-large-xlsr-deepfake-audio-classification (cached)    |
| RawNet2  | Fine-tuned (Kaggle T4)       | 1.000   | finetune/checkpoints/rawnet2_trained_best.pth                          |

### Training Dataset (used for RawNet2 + AASIST3)
```
Real  : LibriSpeech dev-clean (~2700 clips via torchaudio.datasets.LIBRISPEECH)
Fake  : 500 Kokoro + 500 edge-tts = 1000 clips (generated on Kaggle T4)
Split : 70/15/15 train/val/test
Total : ~1400 training samples
```

### Kaggle Notebooks
| File | Purpose |
|------|---------|
| `kaggle_notebook.py` | Original full notebook (all 3 models, ~10 hrs) |
| `kaggle_rawnet2_only.py` | RawNet2 only (~45-60 min) |
| `kaggle_aasist3_only.py` | AASIST3 only (~40-55 min) |

### GPU
- Training: Kaggle T4 (16 GB VRAM, free 30 hrs/week)
- Local: No GPU (laptop only, CPU inference)

---

## Folder Structure
```
VARE/
├── app/                        ← FastAPI server + UI
├── models/                     ← aasist3/ + aasist_reference/
├── finetune/
│   ├── checkpoints/
│   │   ├── rawnet2_trained_best.pth    ← RawNet2 fine-tuned
│   │   ├── aasist3_finetuned.pth       ← AASIST3 fine-tuned
│   │   └── wav2vec2_finetuned_full/    ← (legacy overfit, not used)
│   └── data/                   ← local ElevenLabs samples
├── evaluation/                 ← evaluate.py (EER / t-DCF)
├── vare_env/                   ← local Python venv (not in git)
├── kaggle_notebook.py          ← full training notebook
├── kaggle_rawnet2_only.py      ← RawNet2-only notebook
├── kaggle_aasist3_only.py      ← AASIST3-only notebook
├── KAGGLE_SETUP.md             ← original Kaggle setup guide
├── KAGGLE_RAWNET2_SETUP.md     ← RawNet2-only guide
├── requirements.txt            ← local dependencies
├── setup_env.bat               ← one-click local env setup
├── vare_api.http               ← API reference for integration
├── .gitattributes              ← marks *.pth as binary (prevents CRLF corruption)
└── PROGRESS.md                 ← this file
```

---

## Detection Philosophy

| Score   | Label  | Meaning                              |
|---------|--------|--------------------------------------|
| 0–30%   | LOW    | Passes cleanly — confident real      |
| 31–60%  | MEDIUM | Suspicious — flag for review         |
| 61–100% | HIGH   | Strong synthetic indicators          |

---

## Step Completion Log

| Date       | Step | Status | Notes |
|------------|------|--------|-------|
| 2026-03-24 | 1  | ✅ DONE | PROGRESS.md created |
| 2026-03-25 | 2  | ✅ DONE | Pushed to github.com/tanish07dollar/Vare |
| 2026-03-25 | 3  | ✅ DONE | WaveFake skipped (wrong Zenodo ID). Used LibriSpeech dev-clean instead |
| 2026-03-25 | 4  | ✅ DONE | 1000 Kokoro + 1000 edge-tts fake clips generated on Kaggle T4 |
| 2026-03-25 | 5  | ✅ DONE | dataset.csv built: 2798 train / 599 val / ~600 test |
| 2026-03-25 | 6  | ✅ DONE | RawNet2 trained 30 epochs on Kaggle — session ended before download |
| 2026-03-25 | 7  | ✅ DONE | wav2vec2 fine-tuned 15 epochs. val_acc=1.000 |
| 2026-03-25 | 8  | ✅ DONE | AASIST3 ran 13/20 epochs before session ended — checkpoint lost |
| 2026-03-26 | 9  | ✅ DONE | Local env created (vare_env). All packages installed |
| 2026-03-26 | 10 | ✅ DONE | app.py updated: HuggingFace wav2vec2, AASIST3 fine-tune loader, RawNet2 loader fixed |
| 2026-03-26 | 11 | ✅ DONE | .gitattributes added — fixed CRLF corruption of .pth files on Windows |
| 2026-03-26 | 12 | ✅ DONE | RawNet2 retrained on kaggle_rawnet2_only.py — val_acc=1.000. Checkpoint placed locally |
| 2026-03-27 | 13 | ✅ DONE | AASIST3 fine-tuned on kaggle_aasist3_only.py — val_acc=0.820. Checkpoint placed locally |
| 2026-03-27 | 14 | ✅ DONE | All 3 models loading. Server running at http://127.0.0.1:8001 |
| 2026-03-27 | 15 | ✅ DONE | vare_api.http updated and pushed for colleague integration handoff |

---

## Problems & Solutions

| Date       | Step | Problem | Solution |
|------------|------|---------|----------|
| 2026-03-24 | — | app.py had hardcoded home paths after restructure | Updated to Path(__file__).parent.parent relative paths |
| 2026-03-24 | — | RawNet2 Val Acc 1.0 — overfit on 101 ElevenLabs samples | Root cause: 1 fake source. Fix: retrain on LibriSpeech + Kokoro + edge-tts |
| 2026-03-25 | Cell 5 | edge-tts: asyncio.run() cannot be called from running event loop | Added nest_asyncio.apply() |
| 2026-03-25 | Cell 8 | RawNet2Spoof.py blocked by nested .gitignore in aasist_reference/ | Cloned original AASIST repo on Kaggle as fallback |
| 2026-03-25 | Cell 10 | ModuleNotFoundError: No module named 'model' (aasist3) | sys.path unreliable — used importlib.util.spec_from_file_location to manually register package |
| 2026-03-26 | — | Kaggle session ended — all checkpoints lost from /kaggle/working/ | Rewrote notebooks to save directly to /kaggle/working/ (Output tab). Always available even if session ends |
| 2026-03-26 | — | rawnet2_trained_best.pth + RawNet2.pth corrupt (truncated ZIP) | Root cause: core.autocrlf=true — git converting LF→CRLF in binary files. Fix: .gitattributes with *.pth binary |
| 2026-03-26 | — | torch not installed locally | Created vare_env with setup_env.bat. torch CPU-only install via PyTorch whl index |
| 2026-03-26 | Cell 6 (rawnet2) | RawNet2Spoof.py not found after fresh clone | Added fallback: auto-clone clovaai/aasist if file missing |
| 2026-03-27 | Cell 6 (aasist3) | ModuleNotFoundError: No module named 'model' | importlib manual package registration — same fix as original Cell 10 |
| 2026-03-27 | Cell 2 (aasist3) | kokoro install failed — Internet OFF on new Kaggle GPU | Enable Internet: Settings → Internet → ON before running Cell 2 |
| 2026-03-27 | Cell 6 (aasist3) | CUDA OutOfMemoryError with batch=6 | Reduced to batch=2 + mixed precision (autocast + GradScaler) + gradient accumulation ×4 |

---

## Session Log

### Session 1 — 2026-03-24
- Full codebase exploration and architecture review
- Folder restructuring, updated app.py import paths
- Identified training data problem (1 fake source → overfit)
- Built complete Kaggle training notebook (kaggle_notebook.py)
- Decided dataset: WaveFake (primary) + Kokoro + edge-tts

### Session 2 — 2026-03-25
- Ran all Kaggle cells: dataset downloaded, fakes generated, all 3 models trained
- Session ended before checkpoints downloaded — all lost
- RawNet2 val_acc=?, wav2vec2 val_acc=1.000, AASIST3 reached epoch 13/20

### Session 3 — 2026-03-26 to 2026-03-27
- Diagnosed checkpoint loss (session expiry) and .pth corruption (CRLF/git)
- Fixed .gitattributes, rewrote save strategy (Output tab)
- Created focused single-model notebooks (kaggle_rawnet2_only.py, kaggle_aasist3_only.py)
- Retrained RawNet2 (val_acc=1.000) and AASIST3 (val_acc=0.820) on Kaggle
- Set up local vare_env, installed all dependencies
- Updated app.py: HuggingFace wav2vec2, fine-tuned AASIST3 loader, RawNet2 loader fix
- All 3 models running locally
- Created vare_api.http for colleague integration handoff
