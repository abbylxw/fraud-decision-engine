// Builds the demo video from scenes.html. See README.md in this folder.
//
//   node record.mjs capture [grafana|alerts]   # screenshots of Grafana, Prometheus, Alertmanager -> shots/
//                             # (run scripts/alert_drill.py at the same time)
//   node record.mjs render    # every frame of scenes.html -> OUT/fraud_engine_demo.mp4 + preview.gif
//
// Needs Playwright and ffmpeg, kept outside the project:
//   npm i --prefix "$TOOLS" playwright@1.58.0 ffmpeg-static@5.2.0
//   TOOLS=... OUT=... node record.mjs render
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TOOLS = process.env.TOOLS;
if (!TOOLS) throw new Error("set TOOLS to the folder where playwright and ffmpeg-static are installed");
const require = createRequire(path.join(TOOLS, "package.json"));
const { chromium } = require("playwright");
const FFMPEG = require("ffmpeg-static");
const OUT = process.env.OUT || path.join(HERE, "out");
const FPS = 30;
const sleep = ms => new Promise(r => setTimeout(r, ms));

const shot = name => path.join(HERE, "shots", name);

async function grafana(browser, minutes) {
  const page = await browser.newPage({ viewport: { width: 1740, height: 3400 }, deviceScaleFactor: 1 });
  // Grafana: one picture per dashboard row, dark theme, kiosk mode.
  await page.goto(`http://localhost:3000/d/fraud-engine-v1?orgId=1&from=now-${minutes}m&to=now&theme=dark&kiosk`,
                  { waitUntil: "networkidle" });
  await sleep(6000);
  const rows = {
    "grafana_golden.png": ["1. Latency (p50 / p99)", "Action rates against their 1-day normal (%)"],
    "grafana_decisions.png": ["Decision mix: allow / review / deny (% of decisions)", "Shadow rules: would-deny hits (per second)"],
    "grafana_drift.png": ["Alerts firing or pending", "Review rate: live vs expected at training time (%)"],
  };
  for (const [file, [first, last]] of Object.entries(rows)) {
    const box = await page.evaluate(([a, b]) => {
      const find = t => [...document.querySelectorAll("h2, h6, [title]")].find(e => (e.textContent || "").trim() === t);
      const panel = e => e.closest("[data-viz-panel-key], .react-grid-item, section");
      const r1 = panel(find(a)).getBoundingClientRect(), r2 = panel(find(b)).getBoundingClientRect();
      return { x: 0, y: r1.top - 8, width: 1740, height: r2.bottom - r1.top + 16 };
    }, [first, last]);
    await page.screenshot({ path: shot(file), clip: box });
    console.log("saved", file, Math.round(box.height), "px tall");
  }
}

async function alerts(browser) {

  // Prometheus: wait for the drill's alert to go pending, then firing.
  const state = async () => {
    const r = await fetch("http://localhost:9090/api/v1/alerts").then(r => r.json());
    const a = r.data.alerts.find(a => a.labels.alertname === "FraudSignupBurstAttack");
    return a ? a.state : "inactive";
  };
  const prom = await browser.newPage({ viewport: { width: 1740, height: 640 } });
  for (const want of ["pending", "firing"]) {
    console.log("waiting for", want);
    while ((await state()) !== want) await sleep(3000);
    await prom.goto("http://localhost:9090/alerts?search=FraudSignupBurstAttack", { waitUntil: "networkidle" });
    await prom.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
    const rule = prom.getByText("FraudSignupBurstAttack").first();
    await rule.click().catch(() => {});
    await sleep(1500);
    await prom.screenshot({ path: shot(`prom_${want}.png`) });
    console.log("saved", `prom_${want}.png`);
  }

  // Alertmanager: the alert as people receive it.
  await sleep(20000);
  const am = await browser.newPage({ viewport: { width: 1740, height: 640 } });
  await am.goto("http://localhost:9093/#/alerts?silenced=false&inhibited=false&active=true&filter=%7Balertname%3D%22FraudSignupBurstAttack%22%7D",
                { waitUntil: "networkidle" });
  await sleep(2000);
  await am.getByText("Info").first().click().catch(() => {});
  await sleep(800);
  await am.screenshot({ path: shot("alertmanager.png") });
  console.log("saved alertmanager.png");
}

async function capture(which) {
  mkdirSync(path.join(HERE, "shots"), { recursive: true });
  const browser = await chromium.launch();
  if (which !== "alerts") await grafana(browser, Number(process.env.MINUTES || 20));
  if (which !== "grafana") await alerts(browser);
  await browser.close();
}

async function render() {
  mkdirSync(OUT, { recursive: true });
  const mp4 = path.join(OUT, "fraud_engine_demo.mp4");
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto("file://" + path.join(HERE, "scenes.html"));
  await page.evaluate(() => document.fonts.ready);
  await sleep(1500);
  const scenes = await page.evaluate(() => window.video.scenes);

  const ff = spawn(FFMPEG, ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg",
    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow",
    "-movflags", "+faststart", mp4], { stdio: ["pipe", "inherit", "inherit"] });
  const FADE = 0.35;
  for (let i = 0; i < scenes.length; i++) {
    const { id, dur } = scenes[i];
    await page.evaluate(i => window.video.show(i), i);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForFunction(() => [...document.images].every(im => im.complete));
    const frames = Math.round(dur * FPS);
    for (let f = 0; f < frames; f++) {
      const t = f / FPS;
      const fade = Math.min(1, t / FADE, (dur - t) / FADE);
      await page.evaluate(([t, o]) => { window.video.seek(t); document.body.style.opacity = o; }, [t, Math.max(0, fade)]);
      const buf = await page.screenshot({ type: "jpeg", quality: 92 });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
    }
    console.log(`scene ${id}: ${frames} frames`);
  }
  ff.stdin.end();
  await new Promise((ok, bad) => ff.on("close", c => (c === 0 ? ok() : bad(new Error("ffmpeg " + c)))));
  await browser.close();
  console.log("wrote", mp4);

  // Preview GIF: the architecture scene (packet travelling through the system), 960 px wide.
  const gif = path.join(OUT, "preview.gif");
  const start = scenes[0].dur;   // scene s1 starts after the title
  await new Promise((ok, bad) => spawn(FFMPEG, ["-y", "-loglevel", "error", "-ss", String(start), "-t", "14", "-i", mp4,
    "-vf", "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
    gif], { stdio: "inherit" }).on("close", c => (c === 0 ? ok() : bad(new Error("gif " + c)))));
  console.log("wrote", gif);
}

const mode = process.argv[2];
if (mode === "capture") await capture(process.argv[3]);   // optional: grafana | alerts
else if (mode === "render") await render();
else console.log("usage: node record.mjs capture|render");
