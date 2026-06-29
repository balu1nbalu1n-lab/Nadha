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

app = FastAPI(title="Nada Voice Analysis", version="4.9.3")

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

def classify_signal_type(y, sr):
    """
    Classifies the recording as melody / drone / speech using
    note-plateau-fraction — validated against real recordings:
      Drone (D.m4a, F.m4a, C.m4a):  ~90-100% plateau, robust CV ~0.0003
      Melody (8 Carnatic pieces):   ~35-50% plateau, robust CV 0.30-0.50
      Speech (Speak-2.m4a):         <10% plateau,    robust CV ~0.26

    IMPORTANT: this runs its OWN independent YIN pass with a fixed,
    wide fmin=50Hz — deliberately decoupled from the mode-dependent
    fmin (60 for speaker, 80 for singer) used elsewhere for harmonic
    analysis. We confirmed on three different low drones (C, D, F.m4a)
    that using the mode's fmin here causes YIN to lock onto a wrong
    octave/harmonic when the true fundamental sits close to or below
    that fmin floor — producing wildly unstable plateau/CV numbers
    purely as an artifact of which coaching mode button was clicked,
    not anything about the actual recording. A single fixed, wide
    fmin for classification purposes only avoids that dependency.

    Plateau fraction is far more reliable than raw pitch variance —
    speech can have WIDE pitch range (prosody) without ever holding
    a stable note, while melody holds discrete swara/note pitches.
    Robust (MAD-based) CV is used as a secondary signal because
    raw std/mean is corrupted by YIN octave-jump tracking glitches,
    which we confirmed occur even on genuinely clean drone instruments.
    """
    import numpy as np
    import librosa
    CLASSIFIER_FMIN = 50

    f0_full = librosa.yin(y, fmin=CLASSIFIER_FMIN, fmax=900, sr=sr)
    voiced = f0_full[(f0_full > CLASSIFIER_FMIN * 0.9) & (f0_full < 900)]
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


def detect_mixed_content(values, times, unit="dB"):
    """
    A wide min-max range across sampled windows can mean two different
    things: (a) one continuous, genuinely variable performance (e.g. a
    singer who gradually builds intensity), or (b) two distinct vocal
    behaviours got pooled into the same segment -- e.g. sustained
    higher-register singing interrupted by improvisational drops back to
    lower notes, which is common in real concert recordings (confirmed by
    ear on a real Carnatic recording where a segment's "weak SF zone"
    reading turned out to be a strong high-register passage diluted by
    lower-note fallback windows averaged into the same statistic).

    A single mean/range cannot distinguish these two cases -- they can
    produce an identical mean and an identical spread. This needs the
    actual per-window values: if there's a single dominant gap that splits
    the windows into two well-separated clusters, that's evidence of (b),
    not (a). With only 3-8 windows this can't be a real clustering
    algorithm -- it's a largest-gap heuristic, deliberately conservative
    (requires a big, clean split) so it only fires when the split is
    obvious enough to act on, not on ordinary noisy variation.

    Threshold note: an earlier version compared the gap to the *total*
    spread, which under-fired on a real Behag recording (Part 3) -- a
    visually obvious 4-vs-4 split (weak cluster -6.4 to 3.6, strong
    cluster 20.6 to 33.8, separated by a 17 dB gap) failed a "gap >= 55%
    of total spread" test because each cluster also had its own internal
    spread, inflating the total. Comparing the gap to the spread *within*
    each resulting cluster instead (gap must clearly exceed how spread-out
    either side is on its own) catches this correctly while still
    rejecting a smoothly increasing sequence with no real seam.
    """
    import numpy as np
    n = len(values)
    if n < 4:
        return {"mixed": False, "groups": None}
    order = sorted(range(n), key=lambda i: values[i])
    sv = [values[i] for i in order]
    st = [times[i] for i in order]
    total_spread = sv[-1] - sv[0]
    if total_spread < 1e-6:
        return {"mixed": False, "groups": None}
    gaps = [sv[i+1] - sv[i] for i in range(n - 1)]
    gap_idx = int(np.argmax(gaps))
    biggest_gap = gaps[gap_idx]
    other_gaps = gaps[:gap_idx] + gaps[gap_idx+1:]
    runner_up = max(other_gaps) if other_gaps else 0
    lo_vals, hi_vals = sv[:gap_idx+1], sv[gap_idx+1:]
    if len(lo_vals) < 2 or len(hi_vals) < 2:
        # One side is a single point -- only call it mixed if that point is
        # an extreme, unambiguous outlier (gap far exceeds every other gap).
        if biggest_gap < 3.0 * max(runner_up, 1e-6):
            return {"mixed": False, "groups": None}
    else:
        lo_spread = max(lo_vals) - min(lo_vals)
        hi_spread = max(hi_vals) - min(hi_vals)
        within_max = max(lo_spread, hi_spread, 1e-6)
        # The separating gap must clearly exceed the spread within either
        # resulting group, and still be the standout gap in the sequence.
        if biggest_gap < 1.25 * within_max or biggest_gap < 1.8 * max(runner_up, 1e-6):
            return {"mixed": False, "groups": None}
    lo_times, hi_times = sorted(st[:gap_idx+1]), sorted(st[gap_idx+1:])
    if len(lo_vals) < 1 or len(hi_vals) < 1:
        return {"mixed": False, "groups": None}
    return {
        "mixed": True,
        "groups": [
            {"label": "weaker", "n": len(lo_vals), "mean": round(float(np.mean(lo_vals)), 1),
             "range": [round(min(lo_vals), 1), round(max(lo_vals), 1)],
             "times": lo_times},
            {"label": "stronger", "n": len(hi_vals), "mean": round(float(np.mean(hi_vals)), 1),
             "range": [round(min(hi_vals), 1), round(max(hi_vals), 1)],
             "times": hi_times},
        ],
        "unit": unit,
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

    signal_info = classify_signal_type(y, SR)
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
    avg_db_seg = librosa.amplitude_to_db(np.mean(np.abs(D_seg), axis=1), ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    D_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag_full = np.abs(D_full)
    adb_full = librosa.amplitude_to_db(np.mean(mag_full, axis=1), ref=np.max)
    frms = librosa.amplitude_to_db(np.sqrt(np.mean(mag_full**2, axis=0)), ref=np.max)
    vm = frms > -38
    vltas = librosa.amplitude_to_db(np.mean(mag_full[:, vm], axis=1), ref=np.max) if np.any(vm) else adb_full
    vfrac = float(np.mean(vm))

    # Slope, harmonics, dominant-H, and SF zone strength all stay
    # SEGMENT-based (avg_db_seg) -- confirmed via direct testing that a
    # full-track average spectrum does not have a stable, well-defined
    # slope or SF peak for a melodic recording (the average blends many
    # different notes/vowels, and on duets, two singers). Tested moving
    # these to full-track and found dominant-H changed incorrectly
    # (Kaapi: validated H2 became H4) and slope/SF-strength became MORE
    # sensitive to arbitrary reference-band choice, not less. Reverted.
    avg_db = avg_db_seg

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

    # Compact LTAS curve + the actual harmonic-peak fit line, for the
    # "Spectral regions of interest" panel's slope chart -- the chart
    # must reflect the SAME basis as the reported slope number, which is
    # fit on discrete harmonic peaks (hf/ha above), not a freeform fit on
    # the raw curve. Sending both means the frontend draws the real
    # curve behind the real fitted line, not two inconsistent things.
    ltas_mask = (freqs >= 60) & (freqs <= 8000)
    ltas_freqs_full = freqs[ltas_mask]
    ltas_db_full = avg_db[ltas_mask]
    # Log-spaced downsample to ~120 points -- a smooth envelope doesn't
    # need FFT-bin resolution, and this keeps the payload small.
    n_pts = min(120, len(ltas_freqs_full))
    log_targets = np.logspace(np.log10(60), np.log10(8000), n_pts)
    ltas_idx = np.searchsorted(ltas_freqs_full, log_targets)
    ltas_idx = np.clip(ltas_idx, 0, len(ltas_freqs_full)-1)
    ltas_idx = sorted(set(ltas_idx.tolist()))
    ltas_curve = [[round(float(ltas_freqs_full[i]),1), round(float(ltas_db_full[i]),1)] for i in ltas_idx]
    harmonic_fit_points = [[h["hz"], h["db"]] for h in harmonics]
    fit_line_db = [round(float(np.polyval(fit, np.log2(60))),1), round(float(np.polyval(fit, np.log2(8000))),1)] if len(hf) >= 3 else None

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

    # Center the retention window on the ACTUAL detected peak (sf_hz),
    # not a fixed 2000-4000Hz span. A fixed wide span can straddle both
    # the rising bump and a later steep falloff, which averages into
    # a misleadingly steep line that masks a real, visible peak.
    approach = (freqs >= sf_hz - 500) & (freqs <= sf_hz + 500)
    if np.sum(approach) >= 4:
        sf_zone_trend = round(float(np.polyfit(np.log2(freqs[approach]+1), avg_db[approach], 1)[0]), 2)
    else:
        sf_zone_trend = slope
    # Retention = how much the SF zone resists the overall rolloff.
    # Formula: sf_zone_trend - slope
    #   sf_zone_trend SHALLOWER (less negative) than slope -> POSITIVE retention
    #     = energy holding up in the SF band = bump/peak = good therapeutic signal
    #   sf_zone_trend STEEPER (more negative) than slope -> NEGATIVE retention
    #     = energy falling away faster than the rest = no bump
    sf_retention = round(sf_zone_trend - slope, 2)

    if   sf_retention >  3.5: sf_shape, sf_code = "Sharp concentrated peak", "sharp"
    elif sf_retention >  1.2: sf_shape, sf_code = "Moderate focused peak",   "moderate"
    elif sf_retention > -0.5: sf_shape, sf_code = "Broad plateau",           "plateau"
    else:                     sf_shape, sf_code = "Below trend",             "below"

    # ---- Supplementary multi-window range for slope / SF strength / retention ----
    # The values above (slope, sf_str, sf_hz, dom_h, sf_retention, sf_shape) are
    # UNCHANGED and remain anchored to the single validated steady segment --
    # we tried making the "best window" the representative value and reverted
    # it after confirming it broke dominant-H (Kaapi's validated H2 became H1)
    # by anchoring harmonics to an atypically loud but unrepresentative moment.
    #
    # This block ONLY adds _min/_max range context, sampled across several
    # windows spread through the track (skipping a short lead-in), the same
    # adaptive-sampling pattern already validated for gamaka. It never feeds
    # back into the primary numbers above.
    lead_in = min(3.0 * SR, max(0, len(y) - seg_len))
    n_sf_windows = int(np.clip(round(dur / 15), 3, 8))
    win_starts_sf = np.linspace(lead_in, max(lead_in, len(y) - seg_len), n_sf_windows).astype(int)

    range_slopes, range_sf_strs, range_retentions, range_times = [], [], [], []
    for ws in win_starts_sf:
        seg_y = y[ws:ws + seg_len]
        f0w = librosa.yin(seg_y, fmin=fmin, fmax=900, sr=SR)
        voicedw = f0w[(f0w > fmin*0.9) & (f0w < 900)]
        # Require at least 60% of the window to be voiced, not just a bare
        # 20-frame minimum (which was only ~9% of a 5s window -- a window
        # that's mostly a deliberate musical pause/silence with one brief
        # sung note at the end could otherwise pass and get treated as a
        # representative sample of the voice, skewing the reported range).
        # Musical silence is real and valuable to the performance; we just
        # don't want it sampled AS IF it were a typical vocal moment.
        if len(voicedw) < 20 or (len(voicedw) / len(f0w)) < 0.60:
            continue
        f0_anchor = float(np.median(voicedw))
        Dw = librosa.stft(seg_y, n_fft=N_FFT, hop_length=HOP)
        adbw = librosa.amplitude_to_db(np.mean(np.abs(Dw), axis=1), ref=np.max)

        harms_w = []
        for n in range(1, int(min(freqs[-1], 9000) / max(f0_anchor, 1)) + 1):
            exp = n * f0_anchor
            if exp > freqs[-1]: break
            tol = exp * 0.07
            mask = (freqs >= exp - tol) & (freqs <= exp + tol)
            if not np.any(mask): continue
            idx = np.argmax(adbw[mask])
            harms_w.append((float(freqs[mask][idx]), float(adbw[mask][idx])))
        if len(harms_w) < 3:
            continue
        hfw = np.array([h[0] for h in harms_w])
        haw = np.array([h[1] for h in harms_w])
        slope_w = float(np.polyfit(np.log2(hfw), haw, 1)[0])

        ref_w = (freqs >= 400) & (freqs <= 1500)
        cr_w = np.polyfit(np.log2(freqs[ref_w]), adbw[ref_w], 1)
        sf_mask_w = (freqs >= SF_LOW) & (freqs <= SF_HIGH)
        sf_above_w = adbw[sf_mask_w] - np.polyval(cr_w, np.log2(freqs[sf_mask_w]))
        sf_str_w = float(np.max(sf_above_w))
        sf_hz_w = float(freqs[sf_mask_w][np.argmax(sf_above_w)])

        approach_w = (freqs >= sf_hz_w - 500) & (freqs <= sf_hz_w + 500)
        if np.sum(approach_w) >= 4:
            trend_w = float(np.polyfit(np.log2(freqs[approach_w]+1), adbw[approach_w], 1)[0])
        else:
            trend_w = slope_w
        retention_w = trend_w - slope_w

        range_slopes.append(slope_w)
        range_sf_strs.append(sf_str_w)
        range_retentions.append(retention_w)
        range_times.append(round(float(ws) / SR, 1))

    if range_slopes:
        slope_min, slope_max = round(min(range_slopes), 2), round(max(range_slopes), 2)
        slope_mean = round(float(np.mean(range_slopes)), 2)
        sf_str_min, sf_str_max = round(min(range_sf_strs), 1), round(max(range_sf_strs), 1)
        sf_str_mean = round(float(np.mean(range_sf_strs)), 1)
        retention_min, retention_max = round(min(range_retentions), 2), round(max(range_retentions), 2)
        # Mean, not just min/max -- a single high or low window among 5-8
        # samples can otherwise read as "typical" when it's actually an
        # outlier. The mean shows which side of the range the performance
        # actually leans toward, so a headline number near one edge of the
        # range can be flagged as atypical rather than representative.
        retention_mean = round(float(np.mean(range_retentions)), 2)
        sf_str_mix = detect_mixed_content(range_sf_strs, range_times, unit="dB")
    else:
        slope_min = slope_max = slope_mean = slope
        sf_str_min = sf_str_max = sf_str_mean = sf_str
        retention_min = retention_max = retention_mean = sf_retention
        sf_str_mix = {"mixed": False, "groups": None}
    n_sf_windows_used = len(range_slopes)

    # Known limitation: the retention trend-fit assumes a smooth, continuous
    # spectral envelope around the peak. That assumption holds for sung melody
    # (formants smear harmonics into a continuous-looking bump) but breaks for
    # drone instruments with sparse, sharply isolated harmonics and deep
    # valleys between them — confirmed on D.m4a, F.m4a, and other real drone
    # recordings, where retention reads "Below trend" despite a genuinely
    # strong, sharp peak that sf_str (a single-point peak-vs-rolloff measure,
    # robust to sparse harmonics) correctly captures.
    #
    # When the signal is classified as a drone AND retention disagrees with
    # a strong sf_str reading, trust sf_str and flag the result as such,
    # rather than presenting an inconclusive retention verdict as definitive.
    sf_shape_reliable = True
    if signal_info["signal_type"] == "drone" and sf_code in ("plateau", "below") and sf_str >= 15:
        sf_shape = "Strong peak (sparse harmonics)"
        sf_code = "sparse_strong"
        sf_shape_reliable = False

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

    def gamaka_in_window(f0_window, anchor_f0):
        """Same extent/rate logic as the original single-segment version,
        with one addition: reject octave-jump outlier frames first using
        a robust median-based filter. The original 'pick the steadiest
        window' approach avoided this problem by construction (it only
        ever looked at the one window least likely to contain glitches);
        sampling multiple windows across the track removes that built-in
        protection, so each window needs its own guard against YIN
        occasionally locking onto the wrong octave for a few frames --
        the same failure mode we confirmed on real drone recordings
        earlier in this session (B.m4a tracked correctly only ~49% of
        the time despite being a genuinely steady tone)."""
        vib_rate_w = extent_st_w = 0.0
        med = np.median(f0_window)
        # Reject frames more than one octave from the window's median --
        # real gamaka sweeps are at most a few semitones, never an octave.
        clean = f0_window[(f0_window > med/1.8) & (f0_window < med*1.8)]
        if len(clean) > 20 and len(clean) >= len(f0_window) * 0.7:
            wl = min(11, (len(clean)//2)*2-1)
            if wl >= 3:
                sm = savgol_filter(clean, wl, 2)
                ff = np.abs(np.fft.rfft(sm - np.mean(sm)))
                fq = np.fft.rfftfreq(len(sm), d=1.0/fr_rate)
                vm2 = (fq >= 3) & (fq <= 12)
                if np.any(vm2):
                    vib_rate_w = float(fq[vm2][np.argmax(ff[vm2])])
                    det = clean - uniform_filter1d(clean, size=max(1, int(fr_rate)))
                    ext_hz = float(np.std(det)) * 2
                    if anchor_f0 > ext_hz/2:
                        extent_st_w = float(12 * np.log2((anchor_f0+ext_hz/2)/(anchor_f0-ext_hz/2)))
        return extent_st_w, vib_rate_w

    # Gamaka extent/rate, MIN-MAX ACROSS THE TRACK:
    #
    # The original approach measured gamaka on the SAME 5-second window
    # chosen for having the LEAST pitch variance -- i.e. specifically
    # measuring ornamental movement at the moment picked for having the
    # least movement. That's not a small sampling quirk, it's internally
    # contradictory, and it meant a singer's most expressive gamaka
    # elsewhere in the performance was invisible to the number reported.
    #
    # Fix: sample several 5-second windows spread evenly across the WHOLE
    # track (adaptive count: roughly one window per 15s of audio, capped
    # 3-8 windows so very short or very long recordings both get sensible
    # coverage), measure gamaka in each, and report the full MIN-MAX RANGE
    # rather than a single number from one arbitrary window. The range
    # itself is informative: a singer who is calm in places and highly
    # ornamented in others looks different from one who is uniformly
    # moderate throughout, and a single segment-based number could not
    # distinguish those two cases.
    n_windows = int(np.clip(round(dur / 15), 3, 8))
    lead_in_gk = min(3.0 * SR, max(0, len(y) - seg_len))
    win_starts = np.linspace(lead_in_gk, max(lead_in_gk, len(y) - seg_len), n_windows).astype(int)
    extents, rates = [], []
    for ws in win_starts:
        f0_w = librosa.yin(y[ws:ws+seg_len], fmin=fmin, fmax=900, sr=SR)
        voiced_w = f0_w[(f0_w > fmin*0.9) & (f0_w < 900)]
        # Same voiced-majority guard as the SF zone window sampling: a
        # window that's mostly musical silence with one brief sung note
        # should not be treated as a representative gamaka sample.
        if len(voiced_w) < 20 or (len(voiced_w) / len(f0_w)) < 0.60:
            continue
        anchor = float(np.median(voiced_w))
        ext_w, rate_w = gamaka_in_window(voiced_w, anchor)
        if ext_w > 0:
            extents.append(ext_w)
            rates.append(rate_w)

    if extents:
        gamaka_min, gamaka_max = round(min(extents), 2), round(max(extents), 2)
        gamaka_rate_min, gamaka_rate_max = round(min(rates), 2), round(max(rates), 2)
        # Keep a single representative figure (the max -- the singer's most
        # expressive sampled moment) for any code path that still wants one
        # number, e.g. the existing classification thresholds.
        max_idx = int(np.argmax(extents))
        extent_st, vib_rate = extents[max_idx], rates[max_idx]
    else:
        gamaka_min = gamaka_max = gamaka_rate_min = gamaka_rate_max = 0.0
        extent_st = vib_rate = 0.0

    pitch_iqr = float(np.percentile(voiced,75)-np.percentile(voiced,25)) if len(voiced)>4 else 30.0

    log.info(f"Analysed {filename} | {mode} | F0={f0:.1f}Hz slope={slope} SF={sf_str}dB sf_trend={sf_zone_trend} retention={sf_retention}")

    return {
        "mode": mode, "duration": round(dur,1), "filename": filename,
        "f0": round(f0,1), "note": note, "vtype": vtype,
        "signal_type": signal_info["signal_type"],
        "plateau_fraction": signal_info["plateau_fraction"],
        "robust_cv": signal_info["robust_cv"],
        "n_harmonics": len(harmonics), "h_strong": h_strong, "h_good": h_good,
        "dominant_H": dom_h["H"], "dominant_hz": round(dom_h["hz"],1),
        "slope": slope,
        "ltas_curve": ltas_curve,
        "harmonic_fit_points": harmonic_fit_points,
        "fit_line_endpoints_db": fit_line_db,
        "sf_str": sf_str, "sf_hz": sf_hz, "sf_ltas": sf_ltas, "sf_gap": sf_gap,
        "sf_zone_trend": sf_zone_trend,
        "sf_retention": sf_retention,
        "slope_min": slope_min, "slope_max": slope_max, "slope_mean": slope_mean,
        "sf_str_min": sf_str_min, "sf_str_max": sf_str_max, "sf_str_mean": sf_str_mean,
        "sf_retention_min": retention_min, "sf_retention_max": retention_max, "sf_retention_mean": retention_mean,
        "n_sf_windows": n_sf_windows_used,
        "sf_str_windows": [round(v, 1) for v in range_sf_strs],
        "sf_str_window_times": range_times,
        "sf_str_mixed": sf_str_mix,
        "sf_slope_local": sf_zone_trend,      # legacy alias
        "sf_slope_deviation": -sf_retention,  # legacy alias (old sign convention)
        "sf_shape": sf_shape, "sf_shape_code": sf_code, "sf_shape_reliable": sf_shape_reliable,
        "low_max":   zone(80,   500,  200, 800),
        "mid_max":   zone(500,  2500, 400, 1500),
        # sf_max previously called zone(2500,3500,...) on the FULL-TRACK
        # average spectrum (adb_full) -- a different signal from sf_str
        # (the headline SF metric, computed on the single steadiest
        # window). On every real recording tested, these disagreed,
        # sometimes by 30+ dB (e.g. one file: sf_str=34.7 "Strong" while
        # the old sf_max=4.5 "Developing") -- the headline card and the
        # "Spectral regions of interest" panel were measuring genuinely
        # different things while both claiming to describe "the SF zone".
        # Reusing sf_str here makes the panel agree with the headline by
        # construction, since they're now the same number.
        "sf_max":    sf_str,
        "upper_max": zone(3500, 8000, 800, 2500),
        "hnr": hnr, "vfrac": round(vfrac,3),
        "gamaka": round(extent_st,2), "gamaka_rate": round(vib_rate,2),
        "gamaka_min": gamaka_min, "gamaka_max": gamaka_max,
        "gamaka_rate_min": gamaka_rate_min, "gamaka_rate_max": gamaka_rate_max,
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
    return {"status": "ok", "version": "4.9.3", "api_key_configured": key_set}


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
