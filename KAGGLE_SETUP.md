# VARE — Kaggle Setup Guide
## Complete step-by-step from laptop → trained models

---

## BEFORE YOU START — One-time setup (do this on your laptop)

### Step 1 — Push VARE to GitHub

```bash
# In your VARE project folder (c:\Users\User\Downloads\VARE_Locally)
git remote add origin https://github.com/YOUR_USERNAME/VARE.git
git branch -M main
git push -u origin main
```

If you don't have a GitHub repo yet:
1. Go to github.com → click **+** → **New repository**
2. Name it `VARE`, set it to **Private**, click **Create repository**
3. Run the commands above

### Step 2 — Create a GitHub Personal Access Token

1. github.com → top-right avatar → **Settings**
2. Left sidebar → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
3. Click **Generate new token (classic)**
4. Note: `VARE Kaggle training`
5. Expiration: 30 days
6. Scope: tick **repo** only
7. Click **Generate token** — **copy it immediately** (shown once)

---

## ON KAGGLE

### Step 3 — Create a New Notebook

1. Go to kaggle.com → **Code** → **+ New Notebook**
2. In the right panel → **Settings**:
   - **Accelerator** → select **GPU T4 x1**
   - **Internet** → turn **ON** (needed to clone GitHub + download models)
3. Click **Save** on settings

### Step 4 — Add GitHub Token as a Secret

1. Top menu → **Add-ons** → **Secrets**
2. Click **+ Add a new secret**
   - Name: `GITHUB_TOKEN`
   - Value: paste the token you copied in Step 2
3. Toggle **Notebook access → ON**
4. Click **Save**

### Step 5 — Create Notebook Cells

Open [kaggle_notebook.py](kaggle_notebook.py) from this project.
Each block marked `# CELL N` is a separate cell in Kaggle.

In Kaggle, click **+ Code** to add a new cell, then paste the block.

| Cell | What it does | Time |
|------|-------------|------|
| **Cell 1** | GPU check — confirms T4 is active | instant |
| **Cell 2** | Install packages (transformers, kokoro, edge-tts, zenodo_get) | ~4 min |
| **Cell 3** | Clone your VARE repo from GitHub | ~1 min |
| **Cell 4** | Download WaveFake dataset from Zenodo (~8.8 GB) | ~20 min |
| **Cell 5** | Generate ~2000 fake audio clips using Kokoro + edge-tts | ~40 min |
| **Cell 6** | Build dataset CSV manifest, balance + split 70/15/15 | ~2 min |
| **Cell 7** | Load shared utilities (run before training) | instant |
| **Cell 8** | Train RawNet2 (30 epochs) | ~3-4 hrs |
| **Cell 9** | Fine-tune wav2vec2 (15 epochs) | ~4-5 hrs |
| **Cell 10** | Fine-tune AASIST3 (20 epochs) | ~2-3 hrs |
| **Cell 11** | Print results + zip checkpoints for download | ~1 min |

**Total: ~10-11 hours** (Kaggle session limit is 12 hours)

### Step 6 — Edit Cell 3 Before Running

In Cell 3, change these two lines to match your GitHub:
```python
GITHUB_USER = "YOUR_GITHUB_USERNAME"   # e.g. "tanish123"
GITHUB_REPO = "VARE"                   # leave as VARE unless you named it differently
```

### Step 7 — Run Cells in Order

**Important rules:**
- Run **Cell 1 first** — confirms GPU is active
- Run **Cell 7 before Cell 8/9/10** — it loads utilities all training cells depend on
- If a cell crashes, fix the issue and re-run that cell only (don't restart from Cell 1)
- Kaggle **auto-saves output** — if session expires, your checkpoints are not lost

---

## AFTER TRAINING

### Step 8 — Download Checkpoints

1. In Kaggle notebook → right panel → **Output** tab
2. You will see `vare_checkpoints.zip`
3. Click **Download**

### Step 9 — Replace Local Checkpoints

On your laptop, extract the zip:
```
vare_checkpoints.zip
├── rawnet2_best.pth           → copy to  finetune/checkpoints/
├── wav2vec2_best.pth          → copy to  finetune/checkpoints/
├── wav2vec2_full/             → copy to  finetune/checkpoints/wav2vec2_finetuned_full/
└── aasist3_best.pth           → copy to  finetune/checkpoints/
```

### Step 10 — Test the Server

```bash
cd VARE
python app/app.py
```

Upload test files:
- A real voice recording → should score **LOW (0–30%)**
- An ElevenLabs or AI-generated clip → should score **HIGH (61–100%)**

---

## Expected Results

| Model    | Expected Val Accuracy | Expected EER |
|----------|-----------------------|--------------|
| RawNet2  | 90–95%                | 5–10%        |
| wav2vec2 | 95–98%                | 2–5%         |
| AASIST3  | 95–98%                | 2–5%         |

These are **genuine generalising results** — the model is validated on unseen synthesis systems, not just unseen samples from the same system.

---

## Troubleshooting

**Cell 1: No GPU found**
→ Settings → Accelerator → GPU T4 x1 → Save → re-run Cell 1

**Cell 3: Authentication failed**
→ Check GITHUB_TOKEN secret is set and "Notebook access" is toggled ON

**Cell 4: zenodo_get fails**
→ Re-run the cell — Zenodo downloads are resumable
→ Or manually download WaveFake.zip on your laptop, upload as a Kaggle dataset, and adjust `WAVEFAKE_DIR`

**Cell 9: CUDA out of memory (wav2vec2)**
→ In Cell 9, change `BATCH_W = 2` and re-run

**Session approaching 12hr limit**
→ Stop after the currently running training cell completes
→ The checkpoint is saved automatically
→ Start a new session, run Cell 7 (utilities), then skip to the next training cell

**Kokoro/edge-tts generation fails**
→ Cell 5 has try/except — it will skip the failing tool and continue
→ Training will still work with WaveFake data alone (7 fake systems)
