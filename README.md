# Pulsepoint
# PulsePoint AI – Shorts Generator

## Live App
(https://pulsepoint-weuc8mbhqjfqbfdkdy39fu.streamlit.app/)

## Demo Video (Mandatory)
https://drive.google.com/file/d/1xiHLJTsMMODo5uZO27tZn9FI-hy5K6_7/view



##Result of demo:
https://drive.google.com/file/d/1NGGbwPDx2F6DVfIqXHChg39tEgdq9zBQ/view?usp=sharing

the demo viedo is 8mins long so if u want to check the result click the above link.

## What This Project Does
This project converts long-form videos (podcasts, talks, interviews) into
short-form vertical videos suitable for Reels / Shorts.

Features:
- Automatic selection of ~30 second highlight clips
- Face-centered vertical (9:16) cropping
- Tiny bottom captions for readability
- Generates 5 non-overlapping shorts per video
- Downloadable MP4 + SRT captions
- ## Optional LLM Enhancement
The current deployment uses deterministic multimodal heuristics for clip selection
(text + audio signals), which works out-of-the-box with no API keys. It allows api keys and was used to make demos as well, so for better use, add ur own gemini api key

Optionally, users can enable LLM-based re-ranking (e.g., Gemini) by providing an API key
to further refine clip selection and hooks, as demonstrated in the demo video.


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
