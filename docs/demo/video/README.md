# Demo video

A 2-minute silent video with captions: architecture, what the data showed (the January bot
attack and the device feature), then serving, monitoring and alerting on the running stack.
The MP4 is attached to a GitHub Release, not committed. `../preview.gif` is a 14-second
excerpt (the architecture scene) shown at the top of the main README.

## What is in this folder

| File | What it is |
|---|---|
| `scenes.html` | The eight scenes, drawn at 1920×1080. Open it with `?play` to watch them in a browser. |
| `record.mjs` | Takes the screenshots of the live stack, then renders every frame of `scenes.html` into an MP4 and the preview GIF. |
| `shots/` | Screenshots of Grafana, Prometheus and Alertmanager taken during a real run (2026-11-20 UTC). |

Every number on screen comes from `docs/summary.md`, `docs/feature_ablation.md`,
`app/decisions/model_meta.json` or the monthly fraud counts of `Fraud_Data.csv`. The
terminal scene is real `scripts/demo.py` output, shortened (fewer columns, five of the eight
stories plus the summary line).

## Rebuild

Playwright (a scriptable headless browser) and ffmpeg (a video encoder) are needed only to
build the video, so they are installed in a folder outside the project:

```bash
export TOOLS=/tmp/video-tools
npm i --prefix "$TOOLS" playwright@1.58.0 ffmpeg-static@5.2.0
npx --prefix "$TOOLS" playwright install chromium

# 1. Live stack with traffic, then the screenshots
docker compose up -d
python scripts/demo.py --rps 40 --seconds 600 &        # fills the dashboards
cd docs/demo/video
node record.mjs capture &                               # Grafana now, then waits for the alert
python ../../../scripts/alert_drill.py                  # makes FraudSignupBurstAttack fire
MINUTES=15 node record.mjs capture grafana              # retake Grafana once traffic has run

# 2. Frames -> MP4 and GIF (about 5 to 10 minutes)
OUT=/tmp/video-out node record.mjs render
```

The recorder does not play the page in real time: every animation is a paused Web
Animation, and for each frame it jumps the animations to that moment and takes a screenshot.
The video therefore comes out the same each time, however slow the machine is.
