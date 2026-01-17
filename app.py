
import os, re, json, math, textwrap, subprocess, wave, time
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
import shutil

def get_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

FFMPEG = get_ffmpeg()



# -----------------------------
# Helpers
# -----------------------------
def sh(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:])
    return r.stdout

def extract_audio_wav(video_path, wav_path, sr=16000):
    sh([FFMPEG,"-y","-i",video_path,"-vn","-ac","1","-ar",str(sr),"-c:a","pcm_s16le",wav_path])

def load_wav_mono16(wav_path):
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit PCM wav")
        sr = wf.getframerate()
        n = wf.getnframes()
        audio = wf.readframes(n)
    x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr

def rms_prefix(x): return np.cumsum(x*x)

def rms_between(prefix, sr, start, end):
    i0 = max(0, int(start*sr))
    i1 = min(len(prefix), int(end*sr))
    if i1 <= i0 + 1: return 0.0
    s = float(prefix[i1-1] - (prefix[i0-1] if i0>0 else 0.0))
    return math.sqrt(s/(i1-i0))

from faster_whisper import WhisperModel

@st.cache_resource
def load_fw_model(model_size: str):
    # Streamlit Cloud is CPU-only, so use int8 for speed
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_whisper(wav_path, model_size="small", language=None):
    model = load_fw_model(model_size)

    segments_out = []
    segments_iter, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=3,
        vad_filter=True
    )
    for seg in segments_iter:
        segments_out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip()
        })
    return segments_out


WISDOM_KEYWORDS = [
    "key","important","lesson","learned","truth","mistake","rule","framework","principle",
    "step","steps","how","why","because","never","always","stop","start","avoid","focus",
    "discipline","confidence","mindset","strategy","system","systems"
]
FILLER = ["um","uh","like","you know","kinda","sorta"]

def text_score(s):
    t = s.lower()
    score = sum(1.0 for k in WISDOM_KEYWORDS if k in t)
    if t.endswith((".", "!", "?")): score += 0.5
    for f in FILLER:
        if f in t: score -= 0.4
    n = len(t.split())
    if 8 <= n <= 30: score += 0.9
    elif n < 5: score -= 0.8
    elif n > 45: score -= 0.8
    return score

def looks_like_good_end(text):
    t = text.strip()
    return t.endswith((".", "!", "?")) or t.endswith((".”","!”","?”"))

def build_candidates(segments, prefix, sr, target=30, tol=8, max_cands=25):
    min_len = max(10.0, target - tol)
    max_len = target + tol

    cands = []
    for i in range(len(segments)):
        if not segments[i]["text"]:
            continue
        s0 = segments[i]["start"]
        for j in range(i, len(segments)):
            s1 = segments[j]["end"]
            dur = s1 - s0
            if dur < min_len:
                continue
            if dur > max_len:
                break
            if not looks_like_good_end(segments[j]["text"]):
                continue

            window_text = " ".join(seg["text"] for seg in segments[i:j+1]).strip()
            if not window_text:
                continue

            ts = sum(text_score(seg["text"]) for seg in segments[i:j+1])
            dur_bonus = 1.0 - min(1.0, abs(dur - target)/target)
            energy = rms_between(prefix, sr, s0, s1)
            audio_bonus = math.log(1e-6 + energy)

            score = 0.90*ts + 0.10*audio_bonus + 1.0*dur_bonus
            cands.append({"start": s0, "end": s1, "dur": dur, "score": score, "text": window_text})

    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:max_cands]

def non_overlapping_topk(cands, k=5, min_gap=3.0):
    picks = []
    for c in sorted(cands, key=lambda x: x["score"], reverse=True):
        ok = True
        for p in picks:
            if not (c["end"] + min_gap <= p["start"] or c["start"] >= p["end"] + min_gap):
                ok = False
                break
        if ok:
            picks.append(c)
            if len(picks) >= k:
                break
    return picks

def make_hook_simple(window_text):
    parts = re.split(r"(?<=[.!?])\s+", window_text.strip())
    parts = [p.strip() for p in parts if len(p.split()) >= 5]
    if not parts:
        parts = [window_text.strip()]
    parts.sort(key=lambda s: (abs(len(s.split())-10), -text_score(s)))
    hook = re.sub(r"[^A-Za-z0-9\s'’-]", "", parts[0])
    hook = " ".join(hook.split()[:12])
    return hook.strip() or "Key takeaway"

def sec_to_srt_ts(sec):
    ms = int(round(sec * 1000))
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def wrap_caption(text, width=38, max_lines=2):
    lines = textwrap.wrap(text, width=width)
    if len(lines) <= max_lines: return "\n".join(lines)
    head = lines[:max_lines-1]
    tail = " ".join(lines[max_lines-1:])
    tail = "\n".join(textwrap.wrap(tail, width=width)[:1])
    return "\n".join(head + [tail])

def write_srt(segments, start, end, srt_path):
    kept = []
    for seg in segments:
        a,b = seg["start"], seg["end"]
        if b <= start or a >= end:
            continue
        kept.append({"start": max(a,start)-start, "end": min(b,end)-start, "text": seg["text"].strip()})

    out = []
    idx = 1
    for k in kept:
        if not k["text"]: continue
        out.append(str(idx))
        out.append(f"{sec_to_srt_ts(k['start'])} --> {sec_to_srt_ts(k['end'])}")
        out.append(wrap_caption(k["text"]))
        out.append("")
        idx += 1
    Path(srt_path).write_text("\n".join(out), encoding="utf-8")

def cut_clip(video_path, start, end, out_path):
    dur = max(0.01, end-start)
    sh([FFMPEG,"-y","-ss",f"{start:.3f}","-t",f"{dur:.3f}","-i",video_path,
        "-c:v","libx264","-preset","veryfast","-crf","18","-c:a","aac","-b:a","128k", out_path])

def face_center_crop_vertical_silent(in_mp4, out_mp4, out_size=(720,1280), detect_every=3, alpha=0.85, require_face=True):
    """
    Face-centered crop using OpenCV Haar (works in Colab fast).
    Smooth + headroom so captions don't sit on face.
    """
    cap = cv2.VideoCapture(in_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if W <= 0 or H <= 0:
        raise RuntimeError("Invalid video dimensions")

    out_w, out_h = out_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cx, cy = W/2, H/2
    crop_h = H * 0.70
    seen_face = False
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % detect_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, int(320 * H / W)))

            faces = face_cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2]*b[3])
                sx = W / small.shape[1]
                sy = H / small.shape[0]

                cx_new = (fx + fw/2) * sx
                cy_new = (fy + fh/2) * sy
                face_h = fh * sy

                cx = alpha * cx + (1-alpha) * cx_new
                cy = alpha * cy + (1-alpha) * cy_new
                crop_h = alpha * crop_h + (1-alpha) * max(H*0.55, min(H*0.95, face_h*3.2))
                seen_face = True

        ch = int(round(crop_h))
        cw = int(round(ch * 9 / 16))
        ch = min(ch, H)
        cw = min(cw, W)

        # headroom factor (0.55) keeps face slightly high, makes room for bottom captions
        y0 = int(round(cy - ch * 0.55))
        x0 = int(round(cx - cw / 2))

        x0 = max(0, min(W - cw, x0))
        y0 = max(0, min(H - ch, y0))

        crop = frame[y0:y0+ch, x0:x0+cw]
        crop = cv2.resize(crop, (out_w, out_h))
        writer.write(crop)

        frame_idx += 1

    cap.release()
    writer.release()

    if require_face and not seen_face:
        raise RuntimeError("No face detected in this clip (require_face=True).")

def mux_audio(video_silent, audio_src, out_with_audio):
    sh([FFMPEG,"-y","-i",video_silent,"-i",audio_src,
        "-map","0:v:0","-map","1:a:0",
        "-c:v","libx264","-preset","veryfast","-crf","18",
        "-c:a","aac","-b:a","128k","-shortest", out_with_audio])

def find_font():
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            return p
    return None

def escape_drawtext(t):
    t = t.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'").replace("%", r"\%")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def burn_subs_and_hook(in_mp4, srt_path, out_mp4, hook=None, burn_captions=True, cap_font=16, cap_margin=30, hook_font=40):
    font = find_font()
    srt_abs = os.path.abspath(srt_path)

    # Bottom captions, tiny, with translucent background box
    style = f"FontSize={cap_font},Outline=1,Shadow=0,BorderStyle=3,BackColour=&HAA000000,MarginV={cap_margin},Alignment=2"

    vf_parts = []
    if hook and font:
        hook_esc = escape_drawtext(hook.upper())
        vf_parts.append(
            f"drawtext=fontfile={font}:text='{hook_esc}':x=(w-text_w)/2:y=24:"
            f"fontsize={hook_font}:fontcolor=white:bordercolor=black:borderw=3"
        )
    if burn_captions:
        vf_parts.append(f"subtitles='{srt_abs}':force_style='{style}'")

    vf = ",".join(vf_parts) if vf_parts else "null"
    sh([FFMPEG,"-y","-i",in_mp4,"-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","18","-c:a","copy", out_mp4])

def generate_shorts(video_path, out_dir, n_shorts, target_sec, tol_sec, min_gap, whisper_model,
                    burn_captions=True, cap_font=16, cap_margin=30, hook_font=40,
                    detect_every=3, require_face=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wav_path = str(out / "audio.wav")
    extract_audio_wav(video_path, wav_path, sr=16000)
    x, sr = load_wav_mono16(wav_path)
    prefix = rms_prefix(x)

    segments = transcribe_whisper(wav_path, model_size=whisper_model, language=None)
    cands = build_candidates(segments, prefix, sr, target=target_sec, tol=tol_sec, max_cands=40)
    picks = non_overlapping_topk(cands, k=n_shorts, min_gap=min_gap)

    results = []
    for idx, c in enumerate(picks, 1):
        start, end = c["start"], c["end"]
        hook = make_hook_simple(c["text"])

        raw   = str(out / f"raw_{idx:02d}.mp4")
        srt   = str(out / f"short_{idx:02d}.srt")
        silent= str(out / f"vert_{idx:02d}_silent.mp4")
        vaud  = str(out / f"vert_{idx:02d}_audio.mp4")
        final = str(out / f"short_{idx:02d}.mp4")
        meta  = str(out / f"short_{idx:02d}.json")

        cut_clip(video_path, start, end, raw)
        write_srt(segments, start, end, srt)

        # face centered
        try:
            face_center_crop_vertical_silent(raw, silent, out_size=(720,1280), detect_every=detect_every, require_face=require_face)
        except Exception:
            # fallback center crop (only if require_face=False)
            if require_face:
                raise
            sh([FFMPEG,"-y","-i",raw,"-vf","crop='ih*9/16:ih:(iw-ih*9/16)/2:0',scale=720:1280","-an",
                "-c:v","libx264","-preset","veryfast","-crf","18", silent])

        mux_audio(silent, raw, vaud)
        burn_subs_and_hook(vaud, srt, final, hook=hook, burn_captions=burn_captions, cap_font=cap_font, cap_margin=cap_margin, hook_font=hook_font)

        payload = {
            "start": float(start), "end": float(end),
            "duration": float(end-start),
            "hook": hook,
            "final_mp4": final,
            "captions_srt": srt
        }
        Path(meta).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        results.append(payload)

    return results


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PulsePoint Shorts Generator", layout="wide")

st.title("PulsePoint AI — Shorts Generator (Streamlit)")
st.caption("Upload a long video → generates 5 short vertical reels with face-centered crop + tiny bottom captions.")

with st.sidebar:
    st.header("Settings")
    n_shorts = st.slider("Number of shorts", 1, 5, 3)
    target_sec = st.slider("Target seconds", 15, 60, 30)
    tol_sec = st.slider("Tolerance (± sec)", 2, 15, 8)
    min_gap = st.slider("Min gap between clips (sec)", 0, 15, 3)
    whisper_model = st.selectbox("Whisper model", ["small", "medium"], index=0)


    st.subheader("Captions")
    burn_captions = st.checkbox("Burn captions into video", value=True)
    cap_font = st.slider("Caption font size (tiny ants)", 10, 30, 14)
    cap_margin = st.slider("Caption bottom margin", 10, 160, 30)
    hook_font = st.slider("Hook font size", 24, 60, 40)

    st.subheader("Face crop")
    detect_every = st.slider("Detect face every N frames", 1, 10, 5)

    require_face = st.checkbox("Require face detection (strict)", value=True)

uploaded = st.file_uploader("Upload MP4", type=["mp4", "mov", "mkv"])

if uploaded is None:
    st.info("Upload a video to begin.")
    st.stop()

run_id = str(int(time.time()))
workdir = Path("runs") / run_id
workdir.mkdir(parents=True, exist_ok=True)
video_path = str(workdir / "input.mp4")

with open(video_path, "wb") as f:
    f.write(uploaded.read())

st.success(f"Saved upload: {video_path}")

if st.button("Generate Shorts", type="primary"):
    status = st.status("Running pipeline...", expanded=True)
    try:
        status.write("Generating shorts (this can take a few minutes)...")
        out_dir = str(workdir / "outputs")
        results = generate_shorts(
            video_path=video_path,
            out_dir=out_dir,
            n_shorts=n_shorts,
            target_sec=target_sec,
            tol_sec=tol_sec,
            min_gap=float(min_gap),
            whisper_model=whisper_model,
            burn_captions=burn_captions,
            cap_font=int(cap_font),
            cap_margin=int(cap_margin),
            hook_font=int(hook_font),
            detect_every=int(detect_every),
            require_face=bool(require_face),
        )
        status.update(label="Done ✅", state="complete", expanded=False)

        st.subheader("Results")
        for i, r in enumerate(results, 1):
            st.markdown(f"### Short {i}: {r['duration']:.1f}s — {r['hook']}")
            st.video(r["final_mp4"])
            col1, col2 = st.columns(2)
            with col1:
                with open(r["final_mp4"], "rb") as f:
                    st.download_button(f"Download short_{i:02d}.mp4", f, file_name=f"short_{i:02d}.mp4")
            with col2:
                with open(r["captions_srt"], "rb") as f:
                    st.download_button(f"Download captions_{i:02d}.srt", f, file_name=f"captions_{i:02d}.srt")

    except Exception as e:
        status.update(label="Failed ❌", state="error", expanded=True)
        st.exception(e)


