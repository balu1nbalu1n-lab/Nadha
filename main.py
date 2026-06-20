"""
Nada Voice Analysis - Backend Server v4.5
API key stored as server environment variable - never exposed to browser.
Users just upload and analyse - no configuration needed.

Deploy on Render:
  Build:  pip install -r requirements.txt
  Start:  uvicorn main:app --host 0.0.0.0 --port $PORT
  Env:    ANTHROPIC_API_KEY = your key from console.anthropic.com
"""
import os, tempfile, logging, subprocess
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nada")

app = FastAPI(title="Nada Voice Analysis", version="4.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SR = 22050      # Full precision
N_FFT = 4096    # Full precision FFT - Standard instance (2GB) handles this
HOP = 512       # Standard hop length
SF_LOW, SF_HIGH = 2500, 3500

# API key lives on the server only - never sent to browser
def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(503, "Anthropic API key not configured on server. Set ANTHROPIC_API_KEY environment variable.")
    return key


# ── ACOUSTIC ANALYSIS ─────────────────────────────────────────

def classify_signal_type(f0_full, fmin):
    """
    Classifies the recording as melody / drone / speech using
    note-plateau-fraction — validated against real recordings:
      Drone (D.m4a, F.m4a):        ~90-100% plateau, robust CV ~0.0003
      Melody (8 Carnatic pieces):   ~35-50% plateau, robust CV 0.30-0.50
      Speech (Speak-2.m4a):         <10% plateau,    robust CV ~0.26

    Plateau fraction is far more reliable than raw pitch variance —
    speech can have WIDE pitch range (prosody) without ever holding
    a stable note, while melody holds discrete swara/note pitches.
    Robust (MAD-based) CV is used as a secondary signal because
    raw std/mean is corrupted by YIN octave-jump tracking glitches,
    which we confirmed occur even on genuinely clean drone instruments.
    """
    import numpy as np
    voiced = f0_full[(f0_full > fmin * 0.9) & (f0_full < 900)]
    if len(voiced) < 20:
        return {"signal_type": "unknown", "plateau_fraction": 0.0, "robust_cv": 0.0}

    med = float(np.median(voiced))
    mad = float(np.median(np.abs(voiced - med)))
    robust_cv = (mad * 1.4826) / med if med > 0 else 0.0

    semitones = 12 * np.log2(voiced / med)
    quantized = np.round(semitones)
    runs = []
    current_run = 1
    for i in range(1, len(quantized)):
        if quantized[i] == quantized[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    long_runs = [r for r in runs if r >= 8]
    plateau_fraction = float(sum(long_runs) / len(quantized)) if long_runs else 0.0

    if plateau_fraction > 0.70:
        signal_type = "drone"
    elif plateau_fraction < 0.15:
        signal_type = "speech"
    else:
        signal_type = "melody"

    return {
        "signal_type": signal_type,
        "plateau_fraction": round(plateau_fraction, 3),
        "robust_cv": round(robust_cv, 4),
    }


def analyse(audio_bytes, filename, mode):
    # Import heavy libraries here (lazy load) — keeps startup memory low
    import numpy as np
    import librosa
    from scipy.signal import savgol_filter
    from scipy.ndimage import uniform_filter1d
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Convert to WAV via ffmpeg — fixes MP3 header corruption
    # Limit to 90 seconds to control memory usage
    wav_path = tmp_path.replace(suffix, "_nada.wav")
    load_path = tmp_path
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-i", tmp_path, "-ar", str(SR), "-ac", "1",
             wav_path, "-y", "-loglevel", "error"],
            capture_output=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            load_path = wav_path
        os.unlink(tmp_path)
    except Exception as fe:
        log.warning(f"ffmpeg conversion failed: {fe}, loading directly")

    try:
        y, _ = librosa.load(load_path, sr=SR, mono=True)
    finally:
        for p in [tmp_path, wav_path]:
            if os.path.exists(p):
                try: os.unlink(p)
                except: pass

    dur = len(y) / SR
    fmin = 60 if mode == "speaker" else 80
    f0_full = librosa.yin(y, fmin=fmin, fmax=900, sr=SR)
    voiced = f0_full[(f0_full > fmin * 0.9) & (f0_full < 900)]

    signal_info = classify_signal_type(f0_full, fmin)
    log.info(f"Signal type: {signal_info['signal_type']} "
             f"(plateau={signal_info['plateau_fraction']:.1%}, "
             f"robust_cv={signal_info['robust_cv']:.4f})")

    seg_len = int(5.0 * SR)
    best_s, best_std = 0, np.inf
    for s in range(0, max(1, len(y) - seg_len), SR):
        sg = f0_full[s // HOP:(s + seg_len) // HOP]
        sg = sg[(sg > fmin * 0.9) & (sg < 900)]
        if len(sg) < 20: continue
        std = float(np.std(sg))
        if std < best_std: best_std, best_s = std, s
    segment = y[best_s:best_s + seg_len]

    f0_seg = librosa.yin(segment, fmin=fmin, fmax=900, sr=SR)
    voiced_sg = f0_seg[(f0_seg > fmin * 0.9) & (f0_seg < 900)]
    f0 = float(np.median(voiced_sg)) if len(voiced_sg) > 0 else 180.0

    vtype = ("Deep Bass" if f0 < 100 else "Bass" if f0 < 130 else
             "Baritone" if f0 < 165 else "Tenor" if f0 < 210 else
             "Alto" if f0 < 265 else "Mezzo-Soprano" if f0 < 330 else "Soprano")
    notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    midi = 12 * np.log2(max(f0, 20) / 440.0) + 69
    note = notes[int(round(midi)) % 12] + str(int(round(midi)) // 12 - 1)

    D_seg = librosa.stft(segment, n_fft=N_FFT, hop_length=HOP)
    avg_db = librosa.amplitude_to_db(np.mean(np.abs(D_seg), axis=1), ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    D_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag_full = np.abs(D_full)
    adb_full = librosa.amplitude_to_db(np.mean(mag_full, axis=1), ref=np.max)
    frms = librosa.amplitude_to_db(np.sqrt(np.mean(mag_full**2, axis=0)), ref=np.max)
    vm = frms > -38
    vltas = librosa.amplitude_to_db(np.mean(mag_full[:, vm], axis=1), ref=np.max) if np.any(vm) else adb_full
    vfrac = float(np.mean(vm))

    harmonics = []
    for n in range(1, int(min(freqs[-1], 9000) / max(f0, 1)) + 1):
        exp = n * f0
        if exp > freqs[-1]: break
        tol = exp * 0.07
        mask = (freqs >= exp - tol) & (freqs <= exp + tol)
        if not np.any(mask): continue
        idx = np.argmax(avg_db[mask])
        harmonics.append({"H": n, "hz": round(float(freqs[mask][idx]), 1),
                          "db": round(float(avg_db[mask][idx]), 2)})

    hf = np.array([h["hz"] for h in harmonics])
    ha = np.array([h["db"] for h in harmonics])
    fit = np.polyfit(np.log2(hf), ha, 1) if len(hf) >= 3 else [-8.0, 0.0]
    slope = round(float(fit[0]), 2)

    dom_h = max(harmonics, key=lambda x: x["db"]) if harmonics else {"H":1,"hz":f0,"db":0}
    h_strong = sum(1 for h in harmonics if h["db"] > -30)
    h_good   = sum(1 for h in harmonics if h["db"] > -45)

    ref = (freqs >= 400) & (freqs <= 1500)
    cr = np.polyfit(np.log2(freqs[ref]), avg_db[ref], 1)
    sf_mask  = (freqs >= SF_LOW) & (freqs <= SF_HIGH)
    sf_freqs = freqs[sf_mask]
    sf_amps  = avg_db[sf_mask]
    sf_above = sf_amps - np.polyval(cr, np.log2(sf_freqs))
    sf_str   = round(float(np.max(sf_above)), 1)
    sf_hz    = round(float(sf_freqs[np.argmax(sf_above)]), 1)

    cr2     = np.polyfit(np.log2(freqs[ref]), adb_full[ref], 1)
    sf_ltas = round(float(np.max(adb_full[sf_mask] - np.polyval(cr2, np.log2(sf_freqs)))), 1)
    sf_gap  = round(sf_str - sf_ltas, 1)

    approach = (freqs >= 2000) & (freqs <= 4000)
    if np.sum(approach) >= 4:
        sf_slope_local = round(float(np.polyfit(np.log2(freqs[approach]+1), avg_db[approach], 1)[0]), 2)
    else:
        sf_slope_local = slope
    sf_slope_deviation = round(slope - sf_slope_local, 2)

    if   sf_slope_deviation >  3.5: sf_shape, sf_code = "Sharp concentrated peak", "sharp"
    elif sf_slope_deviation >  1.2: sf_shape, sf_code = "Moderate focused peak",   "moderate"
    elif sf_slope_deviation > -0.5: sf_shape, sf_code = "Broad plateau",           "plateau"
    else:                           sf_shape, sf_code = "Below trend",             "below"

    def zone(flo, fhi, rlo=400, rhi=1500):
        zm = (freqs >= flo) & (freqs <= fhi)
        rm = (freqs >= rlo) & (freqs <= rhi)
        c  = np.polyfit(np.log2(freqs[rm]+1), adb_full[rm], 1)
        return round(float(np.max(adb_full[zm] - np.polyval(c, np.log2(freqs[zm]+1)))), 1)

    fl = int(0.04 * SR)
    hnr_v = []
    for s in range(0, len(segment) - fl, fl // 2):
        fr = segment[s:s+fl]
        if np.max(np.abs(fr)) < 0.01: continue
        acf  = np.correlate(fr, fr, mode="full")[len(fr)-1:]
        acf /= (acf[0] + 1e-10)
        lmin = max(1, int(SR/(f0*1.5)))
        lmax = min(int(SR/(f0*0.5)), len(acf)-1)
        if lmin >= lmax: continue
        r = float(np.clip(np.max(acf[lmin:lmax]), 0.001, 0.9999))
        hnr_v.append(10 * np.log10(r / (1-r)))
    hnr = round(float(np.mean(hnr_v)) if hnr_v else 0.0, 2)

    fr_rate = SR / HOP
    vib_rate = extent_st = 0.0
    if len(voiced_sg) > 20:
        wl = min(11, (len(voiced_sg)//2)*2-1)
        if wl >= 3:
            sm = savgol_filter(voiced_sg, wl, 2)
            ff = np.abs(np.fft.rfft(sm - np.mean(sm)))
            fq = np.fft.rfftfreq(len(sm), d=1.0/fr_rate)
            vm2 = (fq >= 3) & (fq <= 12)
            if np.any(vm2):
                vib_rate = float(fq[vm2][np.argmax(ff[vm2])])
                det = voiced_sg - uniform_filter1d(voiced_sg, size=max(1, int(fr_rate)))
                ext_hz = float(np.std(det)) * 2
                if f0 > ext_hz/2:
                    extent_st = float(12 * np.log2((f0+ext_hz/2)/(f0-ext_hz/2)))

    pitch_iqr = float(np.percentile(voiced,75)-np.percentile(voiced,25)) if len(voiced)>4 else 30.0

    log.info(f"Analysed {filename} | {mode} | F0={f0:.1f}Hz slope={slope} SF={sf_str}dB sfslope={sf_slope_local}")

    return {
        "mode": mode, "duration": round(dur,1), "filename": filename,
        "f0": round(f0,1), "note": note, "vtype": vtype,
        "signal_type": signal_info["signal_type"],
        "plateau_fraction": signal_info["plateau_fraction"],
        "robust_cv": signal_info["robust_cv"],
        "n_harmonics": len(harmonics), "h_strong": h_strong, "h_good": h_good,
        "dominant_H": dom_h["H"], "dominant_hz": round(dom_h["hz"],1),
        "slope": slope,
        "sf_str": sf_str, "sf_hz": sf_hz, "sf_ltas": sf_ltas, "sf_gap": sf_gap,
        "sf_slope_local": sf_slope_local,
        "sf_slope_deviation": sf_slope_deviation,
        "sf_shape": sf_shape, "sf_shape_code": sf_code,
        "low_max":   zone(80,   500,  200, 800),
        "mid_max":   zone(500,  2500, 400, 1500),
        "sf_max":    zone(2500, 3500, 500, 1500),
        "upper_max": zone(3500, 8000, 800, 2500),
        "hnr": hnr, "vfrac": round(vfrac,3),
        "gamaka": round(extent_st,2), "gamaka_rate": round(vib_rate,2),
        "pitch_iqr": round(pitch_iqr,1),
    }


# ── API ENDPOINTS ─────────────────────────────────────────────

@app.post("/api/analyse")
async def analyse_endpoint(file: UploadFile = File(...), mode: str = Form(...)):
    if mode not in ("singer", "speaker"):
        raise HTTPException(400, "mode must be singer or speaker")
    data = await file.read()
    if len(data) < 1000:   raise HTTPException(400, "File too small")
    if len(data) > 60*1024*1024: raise HTTPException(413, "File too large (max 60 MB)")
    try:
        metrics = analyse(data, file.filename or "audio.wav", mode)
        return JSONResponse({"ok": True, "metrics": metrics})
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        raise HTTPException(422, f"Could not analyse audio: {e}")


@app.post("/api/narrative")
async def narrative_endpoint(request: Request):
    body   = await request.json()
    prompt = body.get("prompt", "")
    system = body.get("system", "")
    if not prompt: raise HTTPException(400, "Prompt required")

    api_key = get_api_key()

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "system": system,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
            )
        log.info(f"Claude API response status: {resp.status_code}")
        if not resp.is_success:
            raw = resp.text
            log.error(f"Claude API error body: {raw}")
            try:
                detail = resp.json().get("error", {}).get("message", raw)
            except Exception:
                detail = raw
            raise HTTPException(resp.status_code, detail)
        data = resp.json()
        text = data["content"][0]["text"] if data.get("content") else ""
        return JSONResponse({"ok": True, "text": text})
    except httpx.TimeoutException:
        log.error("Claude API timed out after 90s")
        raise HTTPException(504, "Claude API timed out — please try again")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Narrative error detail: {type(e).__name__}: {e}")
        raise HTTPException(502, f"Claude API error: {type(e).__name__}: {e}")


@app.get("/api/health")
async def health():
    key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {"status": "ok", "version": "4.6.0", "api_key_configured": key_set}


# ── SERVE FRONTEND ────────────────────────────────────────────

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
@app.get("/{path:path}")
async def frontend(path: str = ""):
    index = os.path.join("static", "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"error": "Frontend not found"}, status_code=404)
