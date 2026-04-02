# VARE UI Visualization — Claude Code prompt sequence (SAFE edition)

Run these prompts **in order** in Claude Code. Each prompt has a verification step.
Prompts marked with ⚠️ SCORE CHECK require you to compare model scores before and after.

**BEFORE YOU START:** Upload a test audio file and save the current response JSON somewhere (copy from DevTools Network tab). You'll compare against this baseline after every backend change.

---

## Prompt 0: Context seeding (run this FIRST, every new session)

```
Read the entire project structure. Focus on:
1. `app/app.py` — the FastAPI server, especially the `/analyze` endpoint, `load_and_resample()`, `segment_audio()`, and the three model functions: `run_aasist3()`, `run_wav2vec()`, `run_rawnet2()`
2. `app/index.html` — the frontend UI
3. `models/aasist_reference/models/RawNet2Spoof.py` — RawNet2 model class
4. `models/aasist3/model/full_model.py` — AASIST3 model class

Understand the complete data flow: audio upload → temp file → load_and_resample → segment_audio → 3 model inference functions → aggregate scores → ensemble → risk_level → JSON response → frontend rendering.

Tell me:
- What are the exact function signatures of run_aasist3, run_wav2vec, run_rawnet2?
- What does the current response JSON structure look like?
- What are the attribute names inside the RawNet2Spoof class (specifically: the SincConv layer, residual blocks, FMS modules, and GRU layer)?
- What are the attribute names inside the AASIST3 model (specifically: inference branches, graph attention layers)?

Do NOT make any changes. Just confirm your understanding.
```

**Why:** Every subsequent prompt depends on Claude Code knowing your exact code. If it gets function names or attribute paths wrong, hooks will crash. Review its answer — if it gets any attribute names wrong, correct it before proceeding.

---

## Prompt 1: Backend — preprocessing data extraction (SAFE — no model contact)

```
In `app/app.py`, modify ONLY the `/analyze` endpoint to compute audio preprocessing visualizations. This code must run AFTER `load_and_resample()` and `segment_audio()` but BEFORE any model inference functions are called.

CRITICAL RULES:
- Do NOT modify run_aasist3(), run_wav2vec(), or run_rawnet2() in any way
- Do NOT modify segment_audio() or load_and_resample()
- Do NOT import any new model-related libraries
- Do NOT change any existing keys in the response JSON
- Wrap ALL new code in a try/except block — if anything fails, set preprocessing = None and let the rest of the endpoint continue normally

Add this computation block:

1. **Downsampled waveform** — take the audio tensor (after resampling to 16kHz), downsample to exactly 2000 points using stride-based selection: `audio_np = audio.squeeze().numpy(); stride = max(1, len(audio_np) // 2000); waveform_points = audio_np[::stride][:2000].tolist()`. This is a plain numpy operation, no model involvement.

2. **Mel spectrogram** — use `torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=1024, hop_length=512)` then `torchaudio.transforms.AmplitudeToDB()`. Convert result to numpy, downsample time axis to max 200 columns via stride, then `.tolist()`. This uses only torchaudio signal processing, NOT any model.

3. **MFCC** — use `torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13, melkwargs={"n_mels": 80})`. Same downsampling and `.tolist()` conversion.

4. **Power spectral density** — `fft_result = torch.fft.rfft(audio.squeeze()); psd_values = (torch.abs(fft_result) ** 2); freqs = torch.fft.rfftfreq(audio.shape[-1], d=1.0/16000)`. Take only first 500 bins. Convert both to `.tolist()`.

5. **Segment boundaries** — iterate over segments list, compute `{"index": i, "start_sample": i * 48000, "end_sample": min(i * 48000 + 64600, total_samples), "start_sec": round(start/16000, 2), "end_sec": round(end/16000, 2)}` for each.

6. **Audio metadata** — `{"duration_sec": round(len(audio.squeeze()) / 16000, 2), "sample_rate": 16000, "num_segments": len(segments), "total_samples": len(audio.squeeze())}`.

Add all 6 items under a new top-level key `"preprocessing"` in the response JSON. The existing response keys must remain exactly as they are — you are only ADDING a new sibling key.

If torchaudio.transforms is not already imported, add the import at the top of the file.
```

**✅ Verify:**
1. Run the server, upload your test audio
2. Check DevTools → Network → response JSON
3. Confirm `preprocessing` key exists with all 6 sub-keys populated
4. **Confirm ALL existing response keys are identical to your saved baseline** (scores, aggregate_scores, risk_level, confidence — every value should be unchanged)
5. Test with a very short audio (<1 second) and a longer one (>10 seconds) to check edge cases

---

## Prompt 2: Backend — RawNet2 intermediate extraction ⚠️ SCORE CHECK

```
In `app/app.py`, modify ONLY the `run_rawnet2()` function to passively extract intermediate data using PyTorch forward hooks.

CRITICAL SAFETY RULES:
- Hooks must be READ-ONLY. They must only call `.detach().cpu()` on tensors and store copies. They must NEVER modify any tensor, call `.requires_grad_()`, or write back to the model.
- Every hook must be stored in a list and ALL hooks must be removed via `hook.remove()` in a finally: block after inference, even if inference crashes.
- ALL new code must be inside a try/except. If hook registration or data extraction fails, return the scores normally with internals=None.
- The actual model forward pass line (where you call the model on the input tensor) must remain EXACTLY as it is now. Do not change its arguments, do not wrap it differently.
- Do not modify model weights, model config, or model state in any way.

First, read `models/aasist_reference/models/RawNet2Spoof.py` and find the exact attribute names for:
- The SincConv layer (look for the class attribute that holds it)
- The residual blocks (look for nn.Sequential or nn.ModuleList holding them)
- The FMS sub-module inside each residual block (look for the sigmoid-based channel attention)
- The GRU layer

Then implement these extractions:

1. **SincConv filter frequencies (one-time, no hook needed):**
   After model is loaded (outside the per-segment loop), read the SincConv parameters:
   - Find the learnable parameters that define f1 (low cutoff) and bandwidth
   - Compute f1 = abs(low_param), f2 = f1 + abs(band_param)
   - Store as a list of {"f1": float, "f2": float, "center": float} for all filters
   - This reads frozen weights only — completely safe

2. **FMS channel weights (per-segment, via hook):**
   - Before the forward pass, register a hook on the FMS sigmoid output layer in each residual block
   - The hook function: `def hook_fn(module, input, output): captured_data[name] = output.detach().cpu().squeeze().tolist()`
   - After the forward pass, read the captured data. Keep only the last segment's values.
   - Remove all hooks in a finally: block

3. **GRU hidden state trajectory (per-segment, via hook):**
   - Register a hook on the GRU layer
   - GRU returns (output, h_n). Capture `output` (batch x time x hidden), compute L2 norm per timestep: `output.detach().cpu().squeeze().norm(dim=-1).tolist()`
   - Keep only the last segment's trajectory.
   - Remove hook in the same finally: block

Add the results to the return value under a new key `"rawnet2_internals"`:
{
    "sinc_filters": [...],     # list of 70 {f1, f2, center} objects
    "fms_weights": {...},      # dict: {"block_0": [floats], "block_1": [floats], ...}
    "gru_trajectory": [...]    # list of floats, one per GRU timestep
}

If run_rawnet2 currently returns just a list of scores, change it to return a tuple: (scores_list, internals_dict). Then update the calling code in /analyze to unpack this tuple. Make sure the scores_list is IDENTICAL to what was returned before.

Show me the exact model attribute names you found in RawNet2Spoof.py before writing any hook code.
```

**⚠️ SCORE CHECK — this is critical:**
1. Upload the SAME test audio file you used for baseline
2. Compare `rawnet2` per-segment scores against your saved baseline — **they must be identical to at least 6 decimal places**
3. If ANY score differs, tell Claude Code: "The RawNet2 scores changed after adding hooks. The hooks are interfering with the forward pass. Remove all hook code and restore the function to its original state. Show me what you changed so we can debug."
4. Check `rawnet2_internals` has the 3 expected keys with populated data
5. Test twice to make sure scores are deterministic (they should be with `torch.no_grad()`)

---

## Prompt 3: Backend — Wav2Vec2 intermediate extraction ⚠️ SCORE CHECK

```
In `app/app.py`, modify ONLY the `run_wav2vec()` function to extract intermediate representations.

CRITICAL SAFETY RULES (same as Prompt 2):
- READ-ONLY extraction only. No modifying tensors, model weights, or model config.
- All new code in try/except with internals=None fallback.
- The model forward pass call must produce IDENTICAL output.
- Clean up any hooks in finally: block.

The Wav2Vec2 model is loaded from HuggingFace. First, read the current run_wav2vec() code to understand:
- How is the model loaded? (AutoModelForAudioClassification or similar)
- How is the forward pass called? (what arguments are passed)
- What does the output object look like?

HuggingFace models support `output_attentions=True` and `output_hidden_states=True` as arguments. BUT — there's a subtlety. Some HuggingFace audio classification wrappers don't forward these kwargs to the base model. You need to check if passing these arguments actually returns the extra data or if it gets silently ignored.

SAFE approach:
1. BEFORE the per-segment loop, temporarily set:
   `model.config.output_attentions = True`
   `model.config.output_hidden_states = True`
2. Run the forward pass as before — the output object should now have `.attentions` and `.hidden_states` attributes
3. AFTER the loop, reset:
   `model.config.output_attentions = False`
   `model.config.output_hidden_states = False`

If this approach doesn't work (the output doesn't have attentions), fall back to:
- Just extract the classification logits (which are already available) and skip attention/hidden states
- Return partial internals rather than breaking inference

Extract:
1. **Last-layer attention map** — from `outputs.attentions[-1]`, average across heads (dim=1), squeeze, take the last segment's, downsample to max 50x50, `.detach().cpu().tolist()`

2. **Layer activation norms** — from `outputs.hidden_states` (tuple of 25 tensors for 24 layers + embedding), compute `tensor.detach().cpu().squeeze().norm(dim=-1).mean().item()` for each. Return list of 25 floats.

3. **Top attention frames** — from the last-layer attention (already computed in step 1), sum columns to get per-frame total attention. Take top 10 indices and values. Return as list of `{"frame_index": int, "attention_score": float}`.

Return as `"wav2vec2_internals"` alongside the scores. Same tuple-return pattern as Prompt 2.
```

**⚠️ SCORE CHECK:**
1. Upload the SAME test audio
2. Compare `wav2vec2` per-segment scores against baseline — **must be identical**
3. If scores changed: "The Wav2Vec2 scores changed. Setting output_attentions on the config may have affected the forward pass. Remove all changes to the config and internals extraction, restore the original function."
4. If scores are identical, check `wav2vec2_internals` has data
5. If internals are all null but scores are fine, that's acceptable — we'll note it in the UI

---

## Prompt 4: Backend — AASIST3 intermediate extraction ⚠️ SCORE CHECK

```
In `app/app.py`, modify ONLY the `run_aasist3()` function to extract intermediate data.

CRITICAL SAFETY RULES (same as Prompt 2 and 3):
- READ-ONLY extraction only
- All new code in try/except with internals=None fallback  
- Forward pass must be unchanged
- Clean up hooks in finally: block

First, read `models/aasist3/model/full_model.py` carefully and find:
- Where are the multiple inference branches? (look for parallel paths that produce separate logits)
- Where are the graph attention layers? (look for GAT, GraphAttention, or HtrgGraphPool classes)
- What does the model's forward() method return?

AASIST3 is the most complex model. Be conservative — extract only what's safely accessible:

1. **Branch scores (primary target):**
   - If the model has multiple inference branches that each produce logits, hook into each branch's output
   - Apply softmax to each branch's [bonafide, spoof] logit pair to get probabilities
   - Return as `[[p_bon_1, p_spoof_1], [p_bon_2, p_spoof_2], ...]`
   - If branches aren't separately accessible, extract just the final logits before softmax

2. **Graph attention weights (attempt only if safe):**
   - If you can identify the graph attention layer and it has accessible attention coefficients in its output, hook into it
   - If the layer doesn't cleanly expose attention weights, or if hooking into it is complex and risky, SKIP THIS entirely
   - Return null for graph_attention and add a comment explaining why

3. **Final logits:**
   - The raw [bonafide, spoof] logits before any softmax — these should be directly accessible from the model output
   - Return as `[bonafide_logit, spoof_logit]`

Return as `"aasist3_internals"`:
{
    "branch_scores": [...] or null,
    "graph_attention": [...] or null,  
    "final_logits": [float, float]
}

IMPORTANT: If you're not confident about hooking into the graph attention layers, just return branch_scores and final_logits. Partial data with correct model scores is infinitely better than complete data with broken inference. When in doubt, return null for that field.
```

**⚠️ SCORE CHECK:**
1. Upload the SAME test audio
2. Compare `aasist3` per-segment scores against baseline — **must be identical**
3. If scores changed, roll back immediately
4. It's completely fine if `graph_attention` is null — the UI will handle it gracefully
5. Check that `final_logits` are populated at minimum

---

## ✅ BACKEND COMPLETE — FULL VERIFICATION GATE

Before moving to ANY frontend prompt, do this final check:

```
Upload your test audio file. Compare the COMPLETE response JSON against your original baseline:
- All per-segment scores for all 3 models: MUST be identical
- aggregate_scores: MUST be identical  
- risk_level: MUST be identical
- confidence: MUST be identical
- NEW: preprocessing, rawnet2_internals, wav2vec2_internals, aasist3_internals should exist

If ANYTHING in the original response changed, STOP and debug before continuing.
```

---

## Prompt 5: Frontend — waveform, segments, and audio metadata panel

```
In `app/index.html`, add a NEW visualization section. Do NOT modify or remove any existing HTML elements, CSS classes, or JavaScript functions. Only ADD new code.

Keep the existing dark theme: background #0a0a0f, cards #12121a, borders #1e1e2e, text #e0e0e0, accent cyan #00d4ff.

Add a new `<div id="preprocessing-panel" style="display:none;">` ABOVE the existing results section in the HTML. This panel becomes visible when the /analyze response arrives.

Inside this panel, create:

1. **Audio waveform (Canvas)**
   - Full width of container, height 130px
   - Draw from `response.preprocessing.waveform` (array of ~2000 floats)
   - Waveform color: #00d4ff at 0.7 opacity
   - Behind the waveform, draw segment bands as vertical rectangles with alternating background opacity (odd segments: rgba(0,212,255,0.05), even: rgba(0,212,255,0.1))
   - Below the canvas, a row of small labels: "Seg 1", "Seg 2", etc., positioned at each segment's center
   - Use the segment boundaries from `response.preprocessing.segments` to position the bands

2. **Audio metadata row**
   - 4 small metric cards in a horizontal flex row below the waveform
   - Card 1: Duration (e.g. "4.2s")
   - Card 2: Sample Rate ("16,000 Hz")  
   - Card 3: Segments (e.g. "8")
   - Card 4: Total Samples (e.g. "67,200")
   - Card style: background #12121a, border 1px solid #1e1e2e, border-radius 8px, padding 12px, monospace font for numbers
   - Label in small muted text above, value in larger cyan-accented text below

3. **Show/populate logic**
   - In the existing fetch('/analyze') response handler, BEFORE populating the existing results UI, check if `response.preprocessing` exists
   - If yes, show the preprocessing panel and call a new function `renderPreprocessing(response.preprocessing)` that draws the waveform and fills the cards
   - If no, keep the panel hidden

Canvas drawing function should handle edge cases: empty waveform array, null data, very short arrays (<100 points).

The preprocessing panel should have a subtle fade-in: add CSS transition `opacity 0.3s ease` and toggle opacity from 0 to 1 when showing.
```

**✅ Verify:**
1. Upload audio — preprocessing panel appears with waveform and metadata cards
2. Waveform shows amplitude shape with segment bands behind it
3. Metadata values are populated and formatted correctly
4. Existing results section (gauge, score cards) still renders correctly below
5. Upload a different audio — panel updates correctly (old data replaced)

---

## Prompt 6: Frontend — mel spectrogram and MFCC heatmaps

```
In `app/index.html`, add two heatmap visualizations inside the existing `preprocessing-panel` div, below the metadata cards row. Do NOT modify any existing code — only ADD.

1. **Container layout**
   - Create a flex row with gap 16px that holds two canvas elements side by side
   - Each canvas takes 50% width (use flex: 1), height 180px
   - On screens < 768px, stack them vertically (use flex-wrap: wrap, each child flex-basis: 100%)

2. **Mel spectrogram canvas (left)**
   - Title above: "Mel spectrogram" in 13px muted text (#888)
   - Draw `response.preprocessing.mel_spectrogram` — this is a 2D array (80 rows x N columns)
   - Each cell maps to a color: use a viridis-like scale with 4 stops:
     - Minimum value → #0d0887 (dark blue)
     - 33% → #7e03a8 (purple)  
     - 66% → #f89540 (orange)
     - Maximum value → #f0f921 (yellow)
   - Linear interpolate between stops based on the cell's normalized value (0-1 within the global min/max)
   - Draw using ctx.fillRect() for each cell, scaling cell width = canvas.width / num_columns, cell height = canvas.height / num_rows
   - Y-axis label (left side, rotated): "Mel bands" — or just put a small label above
   - X-axis label (bottom): "Time"

3. **MFCC canvas (right)**
   - Title above: "MFCC coefficients" in 13px muted text
   - Draw `response.preprocessing.mfcc` — 2D array (13 rows x N columns)
   - Color scale is DIVERGING because MFCCs are centered around zero:
     - Minimum (negative) → #2166ac (blue)
     - Zero → #1a1a2e (near-black, matching background)
     - Maximum (positive) → #b2182b (red)
   - Same cell-based drawing approach
   - Y-axis: "Coefficients (0-12)"

4. **Integration**
   - Extend the existing `renderPreprocessing()` function (from Prompt 5) to also call `renderMelSpectrogram()` and `renderMFCC()`
   - If mel_spectrogram or mfcc data is null, show a dark placeholder div with "Not available" text

All canvas rendering must happen AFTER the canvas element is visible in the DOM (use requestAnimationFrame or setTimeout(fn, 0) if needed to ensure dimensions are resolved).
```

**✅ Verify:**
1. Upload audio — two heatmaps appear side by side below metadata cards
2. Mel spectrogram shows colored bands (bright where energy is concentrated, usually in lower mel bands)
3. MFCC shows 13 rows with red/blue coloring
4. Resize browser window below 768px — heatmaps should stack vertically
5. Upload a different audio — heatmaps update (no stale data from previous upload)

---

## Prompt 7: Frontend — frequency spectrum chart

```
In `app/index.html`, add a frequency spectrum visualization below the heatmaps in the preprocessing panel. Do NOT modify existing code.

1. **Canvas element**
   - Full width of container, height 160px
   - Title above: "Frequency spectrum (PSD)" in 13px muted text
   - Background: draw a subtle grid (horizontal and vertical lines in rgba(255,255,255,0.03))

2. **PSD curve**
   - Data: `response.preprocessing.psd.freqs` (Hz) and `response.preprocessing.psd.power`
   - X axis: 0 to 8000 Hz (ignore data above 8kHz). Map to canvas pixel coordinates.
   - Y axis: power values in LOG scale. Use `Math.log10(value + 1e-10)` to avoid log(0). Map min/max log values to canvas top/bottom with 10px padding.
   - Draw as a filled area: 
     - ctx.beginPath(), moveTo(x0, canvasHeight), lineTo for each data point, lineTo(xLast, canvasHeight), closePath()
     - Fill: rgba(0, 212, 255, 0.15)
     - Then draw the top line again as a separate path: stroke with #00d4ff, lineWidth 1.5

3. **Axis ticks and labels**
   - X axis: draw tick marks and labels at 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000 Hz. Labels: "0", "1k", "2k", "3k", "4k", "5k", "6k", "7k", "8k"
   - Y axis: just label "Power (dB)" once on the left side
   - Tick label color: #666, font: 11px monospace

4. **Reference band annotations**
   - Draw faint vertical dashed lines at these frequencies with labels above:
     - 300 Hz: "F0" (fundamental frequency range)
     - 1000 Hz: "F1" (first formant)
     - 3000 Hz: "F2-F3" (formants 2 & 3)
   - Dashed line style: rgba(255,255,255,0.15), lineWidth 1, setLineDash([4, 4])
   - Labels: 10px monospace, rgba(255,255,255,0.3)

5. **Integration**
   - Extend renderPreprocessing() to also call renderPSD()
   - Handle null psd data gracefully

All drawing uses Canvas 2D API only — no external chart libraries.
```

**✅ Verify:**
1. Upload audio — frequency spectrum appears below heatmaps
2. Curve shows clear shape (speech should have peaks in 100-4000Hz range)
3. X-axis labels show Hz values, reference annotations are visible but subtle
4. Grid lines are barely visible (not distracting)

---

## Prompt 8: Frontend — model internals tab container + RawNet2 tab

```
In `app/index.html`, create a new section for model-specific deep visualizations. This goes BELOW the existing results section. Do NOT modify existing code.

1. **Tab container**
   - Create `<div id="model-internals-panel" style="display:none;">`
   - Section title: "Model analysis deep view" in 16px, color #e0e0e0, margin-bottom 16px
   - Tab bar: 3 horizontal tabs labeled "RawNet2", "Wav2Vec2", "AASIST3"
   - Tab styling: inline-block, padding 10px 24px, cursor pointer, font-size 13px, monospace
   - Active tab: color #00d4ff, border-bottom 2px solid #00d4ff
   - Inactive tab: color #666, border-bottom 2px solid transparent, hover color #aaa
   - Below tabs: 3 content divs, only one visible at a time (toggle via JS onclick handlers)
   - Default active tab: RawNet2

2. **RawNet2 tab content — 3 visualizations:**

   a) **SincConv filter bank** (Canvas, full width, height 120px)
      - Title: "Learned SincConv frequency filters"
      - From `rawnet2_internals.sinc_filters` — array of {f1, f2, center} objects
      - X axis: 0 to 8000 Hz
      - Draw each filter as a horizontal rectangle: left edge at f1 Hz position, right edge at f2 Hz position, height ~3px per filter, stacked vertically (70 filters = 70 thin rows)
      - Color: #00d4ff at varying opacity (narrow bands = brighter, wider = dimmer). Opacity = 1.0 - (bandwidth / max_bandwidth) * 0.7
      - If null, show placeholder

   b) **FMS channel weights** (Canvas, full width, height 150px)
      - Title: "FMS channel attention — amplified vs suppressed"
      - From `rawnet2_internals.fms_weights` — pick the last block's weights (find the highest-numbered key)
      - Draw 70 vertical bars, evenly spaced
      - Bar height proportional to the sigmoid value (0 to 1)
      - Bar color: if value > 0.7 → #00ff88 (green, amplified), if 0.3-0.7 → #666 (neutral), if < 0.3 → #ff4444 (red, suppressed)
      - Add a horizontal dashed reference line at 0.5
      - X-axis label: "Channel index (frequency bands)"

   c) **GRU trajectory** (Canvas, full width, height 130px)
      - Title: "GRU hidden state trajectory — suspicion over time"
      - From `rawnet2_internals.gru_trajectory` — array of floats
      - Draw as a line chart. X: timestep index. Y: norm value.
      - Compute mean and mean + 1 standard deviation threshold
      - Line color: #00d4ff where below threshold, transitions to #ff4444 where above threshold (draw two passes — first the full line in cyan, then overdraw segments above threshold in red)
      - Draw the threshold as a horizontal dashed line in rgba(255,100,100,0.3)
      - Label the threshold line: "suspicion threshold"

3. **Show/populate logic**
   - In the fetch response handler, after populating existing results, check if any internals data exists
   - If yes: show model-internals-panel, call renderRawNet2Internals()
   - Tab switching: simple onclick that hides all content divs and shows the clicked one

Handle null internals for each visualization independently — one failing shouldn't prevent others from rendering.
```

**✅ Verify:**
1. Upload audio — model internals panel appears below results
2. RawNet2 tab is active by default, shows 3 visualizations
3. SincConv shows 70 thin horizontal bands across the frequency axis
4. FMS shows 70 vertical bars with green/gray/red coloring
5. GRU trajectory shows a line with red highlights where it spikes
6. Click other tabs — content area should go blank (we haven't built those yet)

---

## Prompt 9: Frontend — Wav2Vec2 tab

```
In `app/index.html`, implement the Wav2Vec2 tab content inside the model-internals-panel. Do NOT modify existing code — add to the Wav2Vec2 content div that was created in Prompt 8.

1. **Attention heatmap** (Canvas, width 300px, height 300px, centered or left-aligned)
   - Title: "Transformer attention map (last layer)"
   - From `wav2vec2_internals.last_layer_attention` — 2D array (max 50x50)
   - Draw as a square heatmap: each cell colored by attention value
   - Color scale: #0a0a0f (zero attention) → #7e03a8 (medium, purple) → #ffffff (high, white)
   - Both axes represent time frames
   - Label X: "Key frames", Y: "Query frames" (small text, 11px, #666)
   - If null, show "Attention maps not available for this model configuration"

2. **Layer activation bars** (Canvas, full width, height 200px)
   - Title: "Transformer layer activations"
   - From `wav2vec2_internals.layer_norms` — array of ~25 floats
   - Draw horizontal bars, one per layer, stacked top to bottom
   - Bar length proportional to norm value (normalized to max)
   - Color: interpolate from #185FA5 (low activation, cool blue) to #EF9F27 (high activation, warm amber)
   - Label each bar on the left: "Layer 1", "Layer 2", ... in 11px monospace #888
   - Show the numeric value at the end of each bar

3. **Top attention frames on mini waveform** (Canvas, full width, height 100px)
   - Title: "Most attended time regions"
   - Draw a miniature version of the waveform (reuse preprocessing.waveform data, same drawing logic as Prompt 5 but smaller and dimmer — #00d4ff at 0.3 opacity)
   - From `wav2vec2_internals.top_attention_frames` — array of {frame_index, attention_score}
   - For each top frame, compute its approximate position on the waveform (frame_index / total_frames * canvas_width)
   - Draw a vertical line at that position: color #ff4444, lineWidth 2, with a small circle at top
   - If there are no top frames data, just show the plain mini waveform

Handle null wav2vec2_internals — show "Wav2Vec2 internals not available" placeholder for the entire tab.
```

**✅ Verify:**
1. Click Wav2Vec2 tab — 3 visualizations render
2. Attention heatmap shows a 2D pattern (often has a bright diagonal)
3. Layer bars show varying activations across layers
4. Mini waveform shows red markers at attended regions

---

## Prompt 10: Frontend — AASIST3 tab

```
In `app/index.html`, implement the AASIST3 tab content. Do NOT modify existing code.

1. **Branch vote chart** (Canvas, full width, height 180px)
   - Title: "Internal branch consensus"
   - From `aasist3_internals.branch_scores` — array of [bonafide_prob, spoof_prob] pairs
   - If we have N branches, draw N groups of 2 bars (grouped bar chart)
   - Each group: green bar for bonafide probability, red bar for spoof probability
   - Group labels: "Branch 1", "Branch 2", etc.
   - Y axis: 0% to 100%
   - If all branches agree (all have spoof > 0.5 or all < 0.5), show a small badge: "Consensus: unanimous"
   - If branches disagree, show: "Consensus: split decision" in amber
   - If branch_scores is null, show a text card explaining: "Branch scores extraction not available. AASIST3 uses 4 parallel inference branches with graph attention — individual branch scores require model architecture hooks."

2. **Graph attention visualization**
   - If `aasist3_internals.graph_attention` is NOT null: render as a heatmap (same approach as Wav2Vec2 attention in Prompt 9, with title "Spectro-temporal graph attention")
   - If it IS null (expected in most cases): show an info card instead:
     - Background: #12121a, border: 1px solid #1e1e2e, border-radius 8px, padding 20px
     - Title: "Graph attention network"
     - Body text (13px, #888): "AASIST3 builds a heterogeneous graph connecting spectral nodes (frequency bands) and temporal nodes (time frames). Graph attention lets each frequency band 'look at' every time frame and vice versa, finding cross-domain artifacts that other models miss. Visualization of these attention weights requires deep model hooks that are not yet implemented."
     - This is informative even without the data

3. **Final logits card**
   - From `aasist3_internals.final_logits` — [bonafide_logit, spoof_logit]
   - Small card showing: "Raw logits — Bonafide: X.XXX | Spoof: X.XXX"
   - Apply softmax to show probabilities too: "Probability — Bonafide: XX.X% | Spoof: XX.X%"
   - Color the spoof probability: green if < 30%, amber if 30-60%, red if > 60%

Handle null aasist3_internals — show placeholder for entire tab.
```

**✅ Verify:**
1. Click AASIST3 tab — content renders
2. Branch chart shows grouped bars (or a descriptive card if branch data isn't available)
3. Graph attention shows either a heatmap or the informative text card
4. Final logits card shows values and colored probability

---

## Prompt 11: Frontend — model agreement matrix + waveform risk overlay

```
In `app/index.html`, add a final analysis section below the model internals tabs. Do NOT modify existing code. Also enhance the waveform from Prompt 5.

1. **Model agreement matrix** (HTML table, not canvas)
   - Title: "Segment × model decision matrix"
   - Create below the model-internals-panel
   - Rows: one per segment (use response.preprocessing.segments for count)
   - Columns: "Segment" | "AASIST3" | "Wav2Vec2" | "RawNet2" | "Ensemble"
   - Cell values: the per-segment scores from the response (these are in the existing response — find them in the scores arrays for each model)
   - Cell background coloring: 
     - score < 0.3 → rgba(0, 255, 100, 0.15) (green tint)
     - score 0.3-0.6 → rgba(255, 200, 0, 0.15) (amber tint)
     - score > 0.6 → rgba(255, 50, 50, 0.15) (red tint)
   - Cell text: show the score as percentage (e.g. "23%")
   - Ensemble column: compute weighted average per segment using the 45/35/20 weights
   - Rows where models DISAGREE (one model < 0.3 and another > 0.6 for the same segment) should have a left border: 3px solid #ffaa00
   - Table styling: match existing theme — background #12121a, border-collapse, cell padding 8px, 12px monospace font, header row in #00d4ff

2. **Waveform risk overlay enhancement**
   - After results arrive, go back to the waveform canvas (from Prompt 5) and overlay risk coloring
   - For each segment, compute its ensemble score (same as above)
   - Draw a semi-transparent overlay rectangle on the waveform:
     - score < 0.3 → rgba(0, 255, 100, 0.12) green
     - score 0.3-0.6 → rgba(255, 200, 0, 0.12) amber
     - score > 0.6 → rgba(255, 50, 50, 0.2) red (slightly more opaque — these are the suspicious parts)
   - This replaces the neutral cyan segment bands from Prompt 5 AFTER results are loaded
   - Create a new function `overlayRiskOnWaveform(segments, scores)` that redraws the overlay on the existing canvas

3. **Show logic**
   - Agreement matrix and risk overlay both populate after the fetch response arrives
   - If per-segment scores are not available (some models returned null), use only the available models' scores and note which model data is missing

The agreement matrix is the most important visualization for your banking demo — it shows at a glance which parts of the audio are suspicious and whether the models agree.
```

**✅ Verify:**
1. Upload audio — waveform now has green/amber/red overlays per segment
2. Below model internals, a table appears showing the decision matrix
3. Disagreement rows have an amber left border
4. Cell colors match the scores
5. Everything still works if one model returned null scores

---

## Prompt 12: Frontend — polish, transitions, error handling

```
Review the ENTIRE `app/index.html` for these issues. Fix each one you find:

1. **Loading states** — when the user clicks "Analyze" and the fetch begins:
   - Immediately show the preprocessing-panel with pulsing placeholder rectangles (CSS animation: background oscillating between #12121a and #1e1e2e over 1.5s) where the waveform, heatmaps, and frequency chart will go
   - When the response arrives, replace placeholders with real canvases
   - The existing loading spinner (if any) should remain — don't remove it

2. **Transition sequencing** — when response arrives, render sections with staggered fade-in:
   - 0ms: preprocessing panel (waveform + metadata)
   - 200ms: heatmaps + frequency chart
   - 400ms: existing results (gauge, scores)
   - 600ms: model internals tabs
   - 800ms: agreement matrix
   - Use CSS opacity transitions (0→1 over 300ms) with JS setTimeout to stagger

3. **Re-upload handling** — if the user uploads a new audio file while results are showing:
   - Hide all visualization panels
   - Clear all canvases (ctx.clearRect)
   - Reset tab state to default (RawNet2)
   - Show loading placeholders again
   - The existing code probably already handles some of this — check and extend it

4. **Null data handling audit** — search for every place in the JavaScript where you access:
   - response.preprocessing.* 
   - response.rawnet2_internals.*
   - response.wav2vec2_internals.*
   - response.aasist3_internals.*
   Add null checks: `if (data && data.field && data.field.length > 0)` before drawing. Each visualization should independently degrade to a "Not available" state.

5. **Canvas resolution** — all Canvas elements should handle high-DPI displays:
   - Set canvas.width = canvas.clientWidth * window.devicePixelRatio
   - Set canvas.height = canvas.clientHeight * window.devicePixelRatio  
   - ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
   - This prevents blurry canvases on Retina/HiDPI screens

6. **Auto-scroll** — after analysis completes and the staggered transitions finish, smooth-scroll to the results section (the gauge). The user sees results first, then can scroll down to explore forensics.

7. **Visual consistency check** — make sure:
   - All backgrounds use #0a0a0f (page), #12121a (cards), #1e1e2e (borders)
   - All accent colors use #00d4ff (cyan) consistently
   - All text uses #e0e0e0 (primary), #888 (secondary), #666 (tertiary)
   - Monospace font for all numbers and data values
   - Border-radius 8px on all cards and panels
   - No element looks out of place with the existing sci-fi aesthetic

Do a final review of the complete file and fix anything that looks broken.
```

**✅ Verify (full end-to-end):**
1. Upload short audio (<3 seconds) — works, single segment
2. Upload medium audio (5-10 seconds) — works, multiple segments
3. Upload long audio (>30 seconds) — works, many segments (check agreement matrix isn't too tall)
4. Re-upload while results are showing — resets cleanly
5. Check all 3 model tabs — each renders or shows appropriate placeholder
6. Resize browser — responsive layouts work
7. **FINAL: compare model scores one more time against your original baseline — they must still be identical**

---

## Troubleshooting

**"Canvas is blank/black"**
→ Tell Claude Code: "The [X] canvas is blank. Check that the canvas element has explicit width/height set in CSS, that the drawing function is called after the element is in the DOM, and that the data array is not empty. Add console.log statements to debug."

**"Scores changed after adding hooks"**
→ Tell Claude Code: "Revert ALL changes to [run_rawnet2/run_wav2vec/run_aasist3]. Restore the function to its exact original state. Then show me the diff of what you changed so I can review it."

**"JSON response is too large / slow"**
→ Tell Claude Code: "The response JSON is too large. Add downsampling: limit waveform to 1000 points, mel spectrogram to 80x100, MFCC to 13x100, PSD to 300 bins, attention maps to 30x30, GRU trajectory to 200 points. Use stride-based downsampling everywhere."

**"HuggingFace model doesn't support output_attentions"**
→ Tell Claude Code: "The wav2vec2 model doesn't return attention maps. Remove the attention extraction code. Keep only layer_norms (from hidden states) and the classification logits. Set last_layer_attention and top_attention_frames to null."

**"AASIST3 hooks crash"**
→ Tell Claude Code: "Remove ALL hook code from run_aasist3. Only extract the final logits from the model output (the return value of the forward pass). Set branch_scores and graph_attention to null. This is the safe fallback."
