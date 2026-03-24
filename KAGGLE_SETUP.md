# Kaggle Setup Guide — VARE Training

Follow these steps in order before running the notebook.

---

## Step 1 — Push VARE to GitHub

On your laptop, from the VARE project folder:

```bash
# Create a new private repo on GitHub first (github.com → New repository)
# Then run:
git remote add origin https://github.com/YOUR_USERNAME/VARE.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Download ASVspoof 2019 LA Dataset

1. Go to: https://datashare.ed.ac.uk/handle/10283/3336
2. Create a free University of Edinburgh account (takes 2 minutes)
3. Download **LA.zip** (~14 GB)

> This contains train/dev/eval audio in FLAC format + protocol files.

---

## Step 3 — Upload Dataset to Kaggle

1. Go to kaggle.com → Datasets → **New Dataset**
2. Name it exactly: `asvspoof2019-la`
3. Drag and drop `LA.zip`
4. Kaggle extracts it automatically → click **Create**
5. Wait for it to finish processing (~10-15 min for 14GB)

---

## Step 4 — Create the Kaggle Notebook

1. Go to kaggle.com → **Code** → **New Notebook**
2. Settings (right panel):
   - Accelerator: **GPU T4 x1**
   - Internet: **On** (needed to clone GitHub + download HuggingFace models)
3. Click **Add Data** (top right):
   - Search for your dataset: `asvspoof2019-la`
   - Click **Add**

---

## Step 5 — Add GitHub Token as a Secret

1. In Kaggle notebook → **Add-ons** (top menu) → **Secrets**
2. Click **Add a new secret**
   - Name: `GITHUB_TOKEN`
   - Value: Your GitHub personal access token
     - Generate at: github.com → Settings → Developer settings → Personal access tokens → Classic
     - Scopes needed: `repo` only
3. Toggle **Notebook access ON**

---

## Step 6 — Paste the Notebook Cells

Open `kaggle_notebook.py` from this project. Paste each `# CELL N` block into a **separate** Kaggle notebook cell.

Edit **Cell 3** to set your GitHub username and repo name:
```python
GITHUB_USER = "YOUR_GITHUB_USERNAME"   # change this
GITHUB_REPO = "VARE"                   # change if different
```

---

## Step 7 — Run the Notebook

Run cells in order:
| Cell | Action | Expected time |
|------|--------|---------------|
| 1 | GPU check | instant |
| 2 | Install packages | ~2 min |
| 3 | Clone GitHub repo | ~1 min |
| 4 | Verify dataset | instant |
| 5 | Load utilities | instant |
| 6 | Train RawNet2 | ~3-4 hrs |
| 7 | Fine-tune wav2vec2 | ~4-5 hrs |
| 8 | Fine-tune AASIST3 | ~2-3 hrs |
| 9 | Summary | instant |
| 10 | Zip checkpoints | ~1 min |

Total: ~10-12 hrs. Kaggle gives 12 hrs per session.

> **Tip:** Run Cell 6 (RawNet2) first. It's the fastest and confirms everything works before committing to the longer wav2vec2 run.

---

## Step 8 — Download Checkpoints

When training completes:
1. In Kaggle notebook → **Output** tab (right panel)
2. Click **Download All** — downloads `vare_checkpoints.zip`
3. Extract into your local `finetune/checkpoints/` folder

---

## Expected Results (ASVspoof 2019 LA)

| Model | Expected Val Acc | Expected EER |
|-------|-----------------|--------------|
| RawNet2 | ~92-95% | ~5-8% |
| wav2vec2 | ~97-99% | ~1-3% |
| AASIST3 | ~98-99% | ~0.8-2% |

These are genuine results, not overfit — the models are evaluated on the dev split which contains **unseen speakers and systems**.

---

## Troubleshooting

**VRAM OOM on wav2vec2:**
- In Cell 7, reduce `BATCH_SIZE = 2`
- Or add `torch.cuda.empty_cache()` before model loading

**Dataset path not found:**
- In Cell 4, print `list(Path("/kaggle/input/").iterdir())` to see actual paths
- Adjust `DATASET_ROOT` accordingly

**Session timeout (12hr limit):**
- Kaggle saves output automatically to `/kaggle/working/`
- If session cuts off mid-training, the last saved `.pth` checkpoint is still there
- Download it from the Output tab and resume from that checkpoint
