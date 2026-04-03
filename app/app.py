import os
import sys
import copy
import json
import tempfile
import traceback
from pathlib import Path
import string
import numpy as np
import torch
import torchaudio

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

try:
    import mutagen
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

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

        # Load fine-tuned weights on top if available
        ckpt_path = Path(__file__).parent.parent / "finetune/checkpoints/aasist3_finetuned.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            m.load_state_dict(state, strict=True)
            val_acc = ckpt.get("val_acc", "?") if isinstance(ckpt, dict) else "?"
            print(f"[VARE] AASIST3 loaded (fine-tuned, val_acc={val_acc})")
        else:
            print("[VARE] AASIST3 loaded (pretrained only)")

        m.eval().to(device)
        return m
    except Exception as e:
        print(f"[VARE] AASIST3 failed to load: {e}")
        return None

def load_wav2vec():
    try:
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        repo = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
        print("[VARE] Loading pretrained wav2vec2 (deepfake-trained)...")
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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Reference weights are a raw state dict; our fine-tuned ones use a wrapper
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            m.load_state_dict(ckpt["model_state"], strict=True)
        else:
            m.load_state_dict(ckpt, strict=True)

        m.eval().to(device)
        print("[VARE] RawNet2 loaded (ASVspoof pretrained)")
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
# AI signature config (metadata analysis)
# ─────────────────────────────────────────────
_SIG_PATH = Path(__file__).parent / "ai_signatures.json"
try:
    with open(_SIG_PATH, encoding="utf-8") as _f:
        AI_SIGNATURES = json.load(_f)
    print(f"[VARE] Loaded AI signatures: {len(AI_SIGNATURES['tag_keywords'])} tag keywords, {len(AI_SIGNATURES['filename_patterns'])} filename patterns")
except Exception as _e:
    AI_SIGNATURES = {"tag_keywords": [], "filename_patterns": []}
    print(f"[VARE] Warning: could not load ai_signatures.json: {_e}")

# ─────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────
TARGET_SR           = 16000
SEGMENT_LEN         = 64600
SEGMENT_HOP         = 48000
MAX_FILE_SIZE_MB    = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DURATION_SEC    = 300   # 5 minutes

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
    mean = sum(scores) / len(scores)
    return 0.6 * mean + 0.4 * max(scores)

def risk_level(score_0_1):
    s = score_0_1 * 100
    if s <= 30:  return "LOW",    "#22c55e"
    if s <= 60:  return "MEDIUM", "#f59e0b"
    return "HIGH", "#ef4444"

# ─────────────────────────────────────────────
# Metadata analysis
# ─────────────────────────────────────────────
def analyze_metadata(file_path: str, original_filename: str) -> dict:
    """
    Check embedded audio tags and filename for known AI TTS tool signatures.
    Returns a metadata_analysis dict. Never raises — falls back to flag_raised=False.
    Does NOT affect the ensemble score.
    """
    no_flag = {"flag_raised": False, "reason": None, "matched_field": None, "matched_value": None}

    tag_keywords      = [k.lower() for k in AI_SIGNATURES.get("tag_keywords", [])]
    filename_patterns = [p.lower() for p in AI_SIGNATURES.get("filename_patterns", [])]

    # ── 1. File tag inspection via mutagen ──────────────────────────────────
    if MUTAGEN_AVAILABLE:
        try:
            audio_file = mutagen.File(file_path, easy=False)
            if audio_file is not None:
                # Build a flat dict of {field_name: string_value} from all tags
                tag_candidates = {}

                # ID3 (MP3): TIT2, COMM, TENC, TSSE, etc.
                if hasattr(audio_file, "tags") and audio_file.tags is not None:
                    for key, val in audio_file.tags.items():
                        key_str = str(key).lower()
                        if any(t in key_str for t in ("enc", "soft", "comm", "tsse", "tenc", "isft", "icmt", "ieng")):
                            tag_candidates[str(key)] = str(val)

                # WAV RIFF INFO chunks (mutagen.wave.WAVE uses _RiffChunk)
                if hasattr(audio_file, "info") and hasattr(audio_file, "tags"):
                    # mutagen exposes RIFF chunks via tags for WAV
                    pass  # already covered by tags loop above

                # FLAC / OGG Vorbis comments — mutagen stores as dict-like
                if hasattr(audio_file, "get"):
                    for vkey in ("encoder", "encoded-by", "comment", "software", "tool"):
                        vals = audio_file.get(vkey, [])
                        if vals:
                            tag_candidates[vkey] = " ".join(str(v) for v in vals)

                # Check each collected tag value against known keywords
                for field, value in tag_candidates.items():
                    value_lower = value.lower()
                    for keyword in tag_keywords:
                        if keyword in value_lower:
                            # Find the original-case keyword for the reason string
                            orig_kw = AI_SIGNATURES["tag_keywords"][tag_keywords.index(keyword)]
                            return {
                                "flag_raised":    True,
                                "reason":         f"{orig_kw} detected in file tags",
                                "matched_field":  field,
                                "matched_value":  value.strip(),
                            }
        except Exception:
            pass  # unreadable tags → fall through to filename check

    # ── 2. Filename pattern check ────────────────────────────────────────────
    name_lower = original_filename.lower() if original_filename else ""
    for pattern in filename_patterns:
        if pattern in name_lower:
            orig_pat = AI_SIGNATURES["filename_patterns"][filename_patterns.index(pattern)]
            return {
                "flag_raised":   True,
                "reason":        f"AI tool pattern '{orig_pat}' detected in filename",
                "matched_field": "filename",
                "matched_value": original_filename,
            }

    return no_flag


# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE_BYTES:
            return JSONResponse(
                {"status": "error", "message": f"File too large ({len(contents) / (1024*1024):.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB."},
                status_code=413
            )

        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        meta_result  = analyze_metadata(tmp_path, file.filename)
        audio        = load_and_resample(tmp_path)
        os.unlink(tmp_path)
        duration_sec = round(audio.shape[1] / TARGET_SR, 1)

        if duration_sec > MAX_DURATION_SEC:
            return JSONResponse(
                {"status": "error", "message": f"Audio too long ({duration_sec}s). Maximum allowed duration is {MAX_DURATION_SEC // 60} minutes."},
                status_code=422
            )

        segments   = segment_audio(audio)
        n_segments = len(segments)

        # ── Preprocessing visualizations (safe — no model contact) ──────────
        preprocessing = None
        try:
            audio_np     = audio.squeeze().numpy()
            total_samples = len(audio_np)

            # 1. Downsampled waveform (~2000 points)
            stride          = max(1, total_samples // 2000)
            waveform_points = audio_np[::stride][:2000].tolist()

            # 2. Mel spectrogram (80 bands, max 200 time columns)
            mel_t  = torchaudio.transforms.MelSpectrogram(
                sample_rate=TARGET_SR, n_mels=80, n_fft=1024, hop_length=512
            )
            db_t   = torchaudio.transforms.AmplitudeToDB()
            mel_np = db_t(mel_t(audio)).squeeze(0).numpy()   # (80, T)
            t_stride = max(1, mel_np.shape[1] // 200)
            mel_list = mel_np[:, ::t_stride][:, :200].tolist()

            # 3. MFCC (13 coefficients, max 200 time columns)
            mfcc_t  = torchaudio.transforms.MFCC(
                sample_rate=TARGET_SR, n_mfcc=13, melkwargs={"n_mels": 80}
            )
            mfcc_np = mfcc_t(audio).squeeze(0).numpy()       # (13, T)
            t_stride2 = max(1, mfcc_np.shape[1] // 200)
            mfcc_list = mfcc_np[:, ::t_stride2][:, :200].tolist()

            # 4. Power spectral density (first 500 bins)
            fft_out    = torch.fft.rfft(audio.squeeze())
            psd_power  = (torch.abs(fft_out) ** 2)
            psd_freqs  = torch.fft.rfftfreq(audio.shape[-1], d=1.0 / TARGET_SR)
            psd_data   = {"freqs": psd_freqs[:500].tolist(), "power": psd_power[:500].tolist()}

            # 5. Segment boundaries
            seg_boundaries = []
            for i in range(n_segments):
                start = i * SEGMENT_HOP
                end   = min(start + SEGMENT_LEN, total_samples)
                seg_boundaries.append({
                    "index":        i,
                    "start_sample": start,
                    "end_sample":   end,
                    "start_sec":    round(start / TARGET_SR, 2),
                    "end_sec":      round(end   / TARGET_SR, 2),
                })

            # 6. Audio metadata
            audio_meta = {
                "duration_sec":  round(total_samples / TARGET_SR, 2),
                "sample_rate":   TARGET_SR,
                "num_segments":  n_segments,
                "total_samples": total_samples,
            }

            # 7. MFCC statistics — mean & std per coefficient + delta
            mfcc_mean        = mfcc_np.mean(axis=1).tolist()    # shape (13,)
            mfcc_std         = mfcc_np.std(axis=1).tolist()     # shape (13,)
            mfcc_delta_np    = np.diff(mfcc_np, axis=1)         # (13, T-1)
            mfcc_delta_mean  = mfcc_delta_np.mean(axis=1).tolist()
            mfcc_delta_std   = mfcc_delta_np.std(axis=1).tolist()
            mfcc_analysis = {
                "mean":       [round(v, 3) for v in mfcc_mean],
                "std":        [round(v, 3) for v in mfcc_std],
                "delta_mean": [round(v, 3) for v in mfcc_delta_mean],
                "delta_std":  [round(v, 3) for v in mfcc_delta_std],
            }

            # 8. Acoustic artifacts
            psd_power_np = np.array(psd_data["power"])
            psd_freqs_np = np.array(psd_data["freqs"])

            # Spectral flatness: geometric mean / arithmetic mean (0=tonal, 1=noise-like)
            geo_mean  = float(np.exp(np.mean(np.log(psd_power_np + 1e-10))))
            arith_mean = float(np.mean(psd_power_np))
            spectral_flatness = round(min(geo_mean / (arith_mean + 1e-10), 1.0), 4)

            # High-frequency energy ratio (energy >4 kHz vs total)
            hf_mask    = psd_freqs_np >= 4000
            hf_ratio   = round(float(psd_power_np[hf_mask].sum() /
                                     (psd_power_np.sum() + 1e-10)), 4)

            # Zero crossing rate per 256-sample frame
            frame_sz   = 256
            n_zcr_frames = max(1, total_samples // frame_sz)
            zcr_rates  = []
            for fi in range(n_zcr_frames):
                frame   = audio_np[fi * frame_sz: (fi + 1) * frame_sz]
                zc      = float(np.sum(np.abs(np.diff(np.sign(frame)))) / 2 / len(frame))
                zcr_rates.append(zc)
            zcr_mean = round(float(np.mean(zcr_rates)), 5)
            zcr_std  = round(float(np.std(zcr_rates)), 5)

            artifact_analysis = {
                "spectral_flatness": spectral_flatness,
                "hf_energy_ratio":   hf_ratio,
                "zcr_mean":          zcr_mean,
                "zcr_std":           zcr_std,
            }

            # 9. Prosody analysis — RMS energy envelope + pause detection
            rms_frame_sz = 512
            n_rms        = max(1, total_samples // rms_frame_sz)
            rms_env      = []
            for fi in range(n_rms):
                frame   = audio_np[fi * rms_frame_sz: (fi + 1) * rms_frame_sz]
                rms_env.append(float(np.sqrt(np.mean(frame ** 2) + 1e-10)))

            peak_rms    = max(rms_env) if rms_env else 1.0
            threshold   = peak_rms * 0.02          # ~−34 dB from peak
            frame_dur   = rms_frame_sz / TARGET_SR

            pause_segs  = []
            in_pause    = False
            pause_start = 0
            for fi, val in enumerate(rms_env):
                if val < threshold and not in_pause:
                    in_pause, pause_start = True, fi
                elif val >= threshold and in_pause:
                    in_pause = False
                    dur = (fi - pause_start) * frame_dur
                    if dur > 0.05:   # ignore sub-50 ms blips
                        pause_segs.append({
                            "start_sec":    round(pause_start * frame_dur, 2),
                            "end_sec":      round(fi * frame_dur, 2),
                            "duration_sec": round(dur, 2),
                        })

            prosody_analysis = {
                "rms_envelope":   [round(v, 6) for v in rms_env],
                "energy_mean":    round(float(np.mean(rms_env)), 6),
                "energy_std":     round(float(np.std(rms_env)), 6),
                "energy_range":   round(float(peak_rms - min(rms_env)), 6),
                "pause_count":    len(pause_segs),
                "pause_segments": pause_segs,
                "avg_pause_dur":  round(float(np.mean([p["duration_sec"] for p in pause_segs])), 3)
                                  if pause_segs else 0.0,
            }

            preprocessing = {
                "waveform":         waveform_points,
                "mel_spectrogram":  mel_list,
                "mfcc":             mfcc_list,
                "psd":              psd_data,
                "segments":         seg_boundaries,
                "audio_metadata":   audio_meta,
                "mfcc_analysis":    mfcc_analysis,
                "artifact_analysis": artifact_analysis,
                "prosody_analysis": prosody_analysis,
            }
        except Exception as prep_err:
            print(f"[VARE] Preprocessing visualization failed: {prep_err}")
            preprocessing = None
        # ────────────────────────────────────────────────────────────────────

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
        if len(active_scores) > 1:
            verdicts   = [1 if s >= 0.5 else 0 for s in active_scores]
            all_agree  = len(set(verdicts)) == 1
            margin     = abs(ensemble - 0.5) * 2
            confidence = round((1.0 if all_agree else 0.4) * 0.5 + margin * 0.5, 3)
        else:
            confidence = 0.5

        return JSONResponse({
            "status":           "ok",
            "filename":         file.filename,
            "metadata_analysis": meta_result,
            "preprocessing":    preprocessing,
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
                    "verdict":              ("AI Generated" if a3_agg >= 0.5 else "Genuine") if a3_agg is not None else None,
                    "segment_scores":       [round(s * 100, 1) for s in a3_scores] if a3_scores else []
                },
                "wav2vec2": {
                    "available":            w2_scores is not None,
                    "error":                w2_err,
                    "aggregate_spoof_prob": round(w2_agg * 100, 1) if w2_agg is not None else None,
                    "verdict":              ("AI Generated" if w2_agg >= 0.5 else "Genuine") if w2_agg is not None else None,
                    "segment_scores":       [round(s * 100, 1) for s in w2_scores] if w2_scores else []
                },
                "rawnet2": {
                    "available":            rn_scores is not None,
                    "error":                rn_err,
                    "aggregate_spoof_prob": round(rn_agg * 100, 1) if rn_agg is not None else None,
                    "verdict":              ("AI Generated" if rn_agg >= 0.5 else "Genuine") if rn_agg is not None else None,
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
    return HTMLResponse(html_path.read_text(encoding='utf-8'))


@app.get("/health")
async def health():
    return {
        "device":        str(device),
        "models_loaded": {k: (v is not None) for k, v in models.items()}
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)