# Pulsepoint
# PulsePoint AI – Shorts Generator

## Live App
PASTE_YOUR_STREAMLIT_CLOUD_LINK_HERE

## Demo Video (Mandatory)
https://drive.google.com/file/d/1xiHLJTsMMODo5uZO27tZn9FI-hy5K6_7/view

## What This Project Does
This project converts long-form videos (podcasts, talks, interviews) into
short-form vertical videos suitable for Reels / Shorts.

Features:
- Automatic selection of ~30 second highlight clips
- Face-centered vertical (9:16) cropping
- Tiny bottom captions for readability
- Generates 5 non-overlapping shorts per video
- Downloadable MP4 + SRT captions

## How It Works
1. Upload a long video
2. Audio is transcribed using Whisper
3. High-quality segments are selected using text + audio signals
4. Face is tracked and centered in vertical crop
5. Captions are burned at the bottom
6. Final short videos are generated

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
