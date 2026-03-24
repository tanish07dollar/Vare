import os
import sys
import copy
import json
import tempfile
import traceback
from pathlib import Path

import torch
import torchaudio
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(title="VARE — Voice Authenticity Risk Engine")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[VARE] Running on: {device}")

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
models = {}

def load_aasist3():
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "models/aasist3"))
        from model import aasist3
        m = aasist3.from_pretrained("MTUCI/AASIST3")
        m.eval().to(device)
        print("[VARE] AASIST3 loaded")
        return m
    except Exception as e:
        print(f"[VARE] AASIST3 failed to load: {e}")
        return None

def load_wav2vec():
    try:
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        finetuned_path = Path(__file__).parent.parent / "finetune/checkpoints/wav2vec2_finetuned_full"
        if finetuned_path.exists():
            repo = str(finetuned_path)
            print("[VARE] Loading fine-tuned wav2vec2...")
        else:
            repo = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
            print("[VARE] Loading pretrained wav2vec2...")
        fe = AutoFeatureExtractor.from_pretrained(repo)
        m  = AutoModelForAudioClassification.from_pretrained(repo)
        m.eval().to(device)
        print("[VARE] wav2vec2 loaded")
        return {"model": m, "extractor": fe}
    except Exception as e:
        print(f"[VARE] wav2vec2 failed to load: {e}")
        return None

def load_rawnet2():
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "models/aasist_reference"))
        from models.RawNet2Spoof import Model

        ckpt_path = Path(__file__).parent.parent / "finetune/checkpoints/rawnet2_trained_best.pth"
        ckpt = torch.load(ckpt_path, map_location=device)

        # Must deepcopy — RawNet2Spoof mutates the config dict during __init__
        correct_config = {
            "nb_samp":      64600,
            "first_conv":   1024,
            "in_channels":  1,
            "filts":        [20, [20, 20], [20, 128], [128, 128]],
            "blocks":       [2, 4],
            "nb_fc_node":   1024,
            "gru_node":     1024,
            "nb_gru_layer": 3,
            "nb_classes":   2
        }
        m = Model(copy.deepcopy(correct_config))
        m.load_state_dict(ckpt["model_state"], strict=True)
        m.eval().to(device)
        print(f"[VARE] RawNet2 loaded (trained, val_acc={ckpt.get('val_acc', '?')})")
        return m
    except Exception as e:
        print(f"[VARE] RawNet2 failed to load: {e}")
        return None

print("[VARE] Loading models...")
models["aasist3"] = load_aasist3()
models["wav2vec2"] = load_wav2vec()
models["rawnet2"]  = load_rawnet2()
print("[VARE] Model loading complete")

# ─────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────
TARGET_SR   = 16000
SEGMENT_LEN = 64600
SEGMENT_HOP = 48000

def load_and_resample(path: str):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != TARGET_SR:
        audio = torchaudio.transforms.Resample(sr, TARGET_SR)(audio)
    return audio

def segment_audio(audio):
    segments = []
    total = audio.shape[1]
    if total <= SEGMENT_LEN:
        pad = SEGMENT_LEN - total
        segments.append(torch.nn.functional.pad(audio, (0, pad)))
    else:
        for start in range(0, total - SEGMENT_LEN + 1, SEGMENT_HOP):
            segments.append(audio[:, start:start + SEGMENT_LEN])
    return segments

# ─────────────────────────────────────────────
# Per-model inference
# ─────────────────────────────────────────────
def run_aasist3(segments):
    m = models["aasist3"]
    if m is None:
        return None, "Model not loaded"
    scores = []
    for seg in segments:
        seg = seg.to(device)
        with torch.no_grad():
            out  = m(seg)
            prob = torch.softmax(out, dim=1)
            scores.append(prob[0][1].item())
    return scores, None

def run_wav2vec(segments):
    bundle = models["wav2vec2"]
    if bundle is None:
        return None, "Model not loaded"
    m  = bundle["model"]
    fe = bundle["extractor"]
    scores = []
    for seg in segments:
        waveform = seg.squeeze(0).cpu().numpy()
        inputs = fe(waveform, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = m(**inputs).logits
            prob   = torch.softmax(logits, dim=-1)
            scores.append(prob[0][1].item())
    return scores, None

def run_rawnet2(segments):
    m = models["rawnet2"]
    if m is None:
        return None, "Model not loaded"
    scores = []
    for seg in segments:
        seg = seg.squeeze(0).unsqueeze(0).to(device)  # (1, 64600)
        with torch.no_grad():
            _, out = m(seg)  # RawNet2Spoof returns (last_hidden, logits)
            prob   = torch.softmax(out, dim=1)
            scores.append(prob[0][1].item())
    return scores, None

# ─────────────────────────────────────────────
# Ensemble aggregation
# ─────────────────────────────────────────────
def aggregate(scores):
    if not scores:
        return 0.0
    weights = [1 + i * 0.1 for i in range(len(scores))]
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

def risk_level(score_0_1):
    s = score_0_1 * 100
    if s <= 30:  return "LOW",    "#22c55e"
    if s <= 60:  return "MEDIUM", "#f59e0b"
    return "HIGH", "#ef4444"

# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        audio        = load_and_resample(tmp_path)
        os.unlink(tmp_path)
        segments     = segment_audio(audio)
        n_segments   = len(segments)
        duration_sec = round(audio.shape[1] / TARGET_SR, 1)

        a3_scores, a3_err = run_aasist3(segments)
        w2_scores, w2_err = run_wav2vec(segments)
        rn_scores, rn_err = run_rawnet2(segments)

        a3_agg = aggregate(a3_scores) if a3_scores else None
        w2_agg = aggregate(w2_scores) if w2_scores else None
        rn_agg = aggregate(rn_scores) if rn_scores else None

        active = [(s, w) for s, w in [(a3_agg, 0.20), (w2_agg, 0.45), (rn_agg, 0.35)] if s is not None]
        if active:
            total_w  = sum(w for _, w in active)
            ensemble = sum(s * w for s, w in active) / total_w
        else:
            ensemble = 0.0

        level, color  = risk_level(ensemble)
        active_scores = [s for s, _ in active]
        confidence    = round(1.0 - float(np.std(active_scores)), 3) if len(active_scores) > 1 else 0.5

        return JSONResponse({
            "status":           "ok",
            "filename":         file.filename,
            "duration_sec":     duration_sec,
            "n_segments":       n_segments,
            "ensemble_score":   round(ensemble * 100, 1),
            "risk_level":       level,
            "risk_color":       color,
            "model_confidence": confidence,
            "models": {
                "aasist3": {
                    "available":            a3_scores is not None,
                    "error":                a3_err,
                    "aggregate_spoof_prob": round(a3_agg * 100, 1) if a3_agg is not None else None,
                    "segment_scores":       [round(s * 100, 1) for s in a3_scores] if a3_scores else []
                },
                "wav2vec2": {
                    "available":            w2_scores is not None,
                    "error":                w2_err,
                    "aggregate_spoof_prob": round(w2_agg * 100, 1) if w2_agg is not None else None,
                    "segment_scores":       [round(s * 100, 1) for s in w2_scores] if w2_scores else []
                },
                "rawnet2": {
                    "available":            rn_scores is not None,
                    "error":                rn_err,
                    "aggregate_spoof_prob": round(rn_agg * 100, 1) if rn_agg is not None else None,
                    "segment_scores":       [round(s * 100, 1) for s in rn_scores] if rn_scores else []
                }
            }
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/health")
async def health():
    return {
        "device":        str(device),
        "models_loaded": {k: (v is not None) for k, v in models.items()}
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)