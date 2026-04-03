# VARE — RawNet2 Only Training (Fresh Guide)
## Train only RawNet2 on Kaggle and download the checkpoint

**Use this guide** when you only need to retrain RawNet2 (e.g. after a lost session).
**Notebook file:** `kaggle_rawnet2_only.py`
**Total time:** ~45–60 minutes

---

## STEP 1 — Push latest code to GitHub (on your laptop)

```bash
cd c:\Users\User\Downloads\VARE_Locally

git add .gitattributes kaggle_rawnet2_only.py KAGGLE_RAWNET2_SETUP.md
git commit -m "Add RawNet2-only notebook and binary gitattributes"
git push
```

> **Why .gitattributes?**
> Without it, git on Windows corrupts `.pth` binary files by converting line endings.
> This is what caused the original checkpoint corruption. Now fixed permanently.

---

## STEP 2 — Create a New Kaggle Notebook

1. Go to **kaggle.com → Code → + New Notebook**
2. Right panel → **Settings**:
   - **Accelerator** → `GPU T4 x1`
   - **Internet** → `ON`
3. Click **Save**

---

## STEP 3 — Add GitHub Token Secret (skip if already added)

1. Top menu → **Add-ons → Secrets → + Add a new secret**
   - Name: `GITHUB_TOKEN`
   - Value: your GitHub personal access token *(repo scope)*
2. Toggle **Notebook access → ON**
3. Click **Save**

> Don't have a token? GitHub → Settings → Developer settings →
> Personal access tokens → Tokens (classic) → Generate new token → tick **repo** → copy it

---

## STEP 4 — Create 6 Cells in Kaggle

Open `kaggle_rawnet2_only.py`. Each block marked `# CELL N` = one Kaggle cell.
Click **+ Code** to add a cell, paste the block, repeat.

| Cell | What it does | Time |
|------|-------------|------|
| **Cell 1** | GPU check — confirms T4 is active | instant |
| **Cell 2** | Install packages (kokoro, edge-tts, soundfile, etc.) | ~3 min |
| **Cell 3** | Clone VARE repo from GitHub | ~1 min |
| **Cell 4** | Download LibriSpeech real audio + generate 500 Kokoro + 500 edge-tts fake clips | ~20 min |
| **Cell 5** | Build dataset CSV manifest | ~1 min |
| **Cell 6** | Train RawNet2 (20 epochs) + auto-save checkpoint to Output tab | ~25–35 min |

---

## STEP 5 — Edit Cell 3 (GitHub username)

Cell 3 has these two lines — they are already set correctly, just verify:

```python
GITHUB_USER = "tanish07dollar"
GITHUB_REPO = "Vare"
```

Change only if your GitHub username or repo name is different.

---

## STEP 6 — Run Cells in Order

Run **Cell 1 → 2 → 3 → 4 → 5 → 6** in order.

**Important:**
- Cell 6 saves `rawnet2_best.pth` directly to `/kaggle/working/` — it appears in the **Output tab automatically**
- The checkpoint is saved **after every epoch that improves val_acc** — even if the session ends early, the best so far is always safe in the Output tab
- Do NOT close or refresh the Kaggle page while Cell 6 is running

---

## STEP 7 — Download the Checkpoint

1. Kaggle notebook → right panel → **Output tab**
2. Find `rawnet2_best.pth`
3. Click the **download icon** next to it

---

## STEP 8 — Replace Local Checkpoint

Copy the downloaded file to your VARE project:

```
rawnet2_best.pth  →  finetune/checkpoints/rawnet2_trained_best.pth
```

In PowerShell:
```powershell
copy "$env:USERPROFILE\Downloads\rawnet2_best.pth" "c:\Users\User\Downloads\VARE_Locally\finetune\checkpoints\rawnet2_trained_best.pth"
```

---

## STEP 9 — Restart the Server

```powershell
# In your VARE_Locally folder with vare_env activated:
vare_env\Scripts\Activate.ps1
python app\app.py
```

All 3 models should now load:
```
[VARE] AASIST3 loaded
[VARE] wav2vec2 loaded
[VARE] RawNet2 loaded (val_acc=0.XXX)
```

---

## Expected Output from Cell 6

```
Ep 01/20 | loss 0.6821 | train_acc 0.541 | val_acc 0.612 | recall 0.580 | prec 0.634
  ✓ Best saved → /kaggle/working/rawnet2_best.pth  (val_acc=0.612)
Ep 02/20 | loss 0.5934 | train_acc 0.703 | val_acc 0.751 | recall 0.720 | prec 0.771
  ✓ Best saved → /kaggle/working/rawnet2_best.pth  (val_acc=0.751)
...
Ep 20/20 | loss 0.3012 | train_acc 0.921 | val_acc 0.883 | recall 0.901 | prec 0.871

Done. Best val_acc: 0.883
Checkpoint: /kaggle/working/rawnet2_best.pth
Checkpoint size: 61.3 MB

Download: Kaggle Output tab (right panel) → rawnet2_best.pth
Then put it in: finetune/checkpoints/rawnet2_trained_best.pth
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Cell 1: No GPU found | Settings → Accelerator → GPU T4 x1 → Save → re-run Cell 1 |
| Cell 3: Authentication failed | Check GITHUB_TOKEN secret is set with "Notebook access" ON |
| Cell 4: Kokoro fails | It's wrapped in try/except — training continues with edge-tts only |
| Cell 4: edge-tts fails | It's wrapped in try/except — training continues with Kokoro only |
| Cell 6: CUDA out of memory | Change `BATCH = 24` to `BATCH = 12` at the top of Cell 6 |
| Output tab empty after session ends | The checkpoint saves to `/kaggle/working/` which IS the Output tab — it's there |
| rawnet2_best.pth not in Output tab | It only appears after Cell 6 saves the first best model (epoch 1+) |
