# Cloud Anime Shorts Factory

Zero-cost cloud pipeline that autonomously generates and publishes one anime-style YouTube Short per day — no Windows PC needed.

## Architecture

```
GitHub Actions (cron 12:00 UTC) — one upload per day
    └── Python Orchestrator
         ├── Scene plan     Cloudflare llama-3.3-70b → OpenRouter → local template
         │                  shot list: size/camera/place/action/object/role/character/narration
         │                  1 part per short, or up to 4 parts queued across days
         ├── Images         strategy `edit`: first character shot = canon, later character
         │                  shots = edits of it; env/insert shots render fresh
         │                  fallbacks: `sheet` (one sheet, cropped) → `fresh` (unlocked)
         ├── Narration      Kokoro-82M TTS (local ONNX), voice af_heart
         ├── Render         each still held for its own line, Ken Burns, 1080x1920, AAC
         └── Upload         YouTube Data API, private first, manual publish
```

No object storage: the rendered MP4 goes straight to YouTube in the same run.

### Shot list (why it reads as a story, not a slideshow)

`scenes.py` plans shots, not narration lines. Every cut carries a job:

| Field | Values |
|---|---|
`shot_size` | establishing, wide, medium_wide, medium, medium_close, close_up, over_shoulder, insert
`camera` | static, pan/tilt, dolly, crane, handheld, zoom …
`narrative_role` | establish_context, introduce_subject, build_tension, deliver_payload, emotional_beat, evidence, resolution, call_to_action
`place` / `action` / `story_object` | concrete and physical — a painter can draw them
`character` | false for inserts and establishing shots, so most cuts never need a face

45 s → **13 shots ≈ 3.3 s per cut**, inside the 1–3 s short-form change rate, with ≥ 6 shots that
have no character on screen. Prompts put **scene first** and the character signature last — a
leading character paragraph collapses every frame into the same portrait (measured: 7 medium
portraits vs. 13 varied shots).

### Character consistency (why the pictures hold together)

1. **Art direction first.** Identity never depends on the face: hooded black raincoat, red scarf,
   dark bob, paper lantern. Most cuts are places, objects and backs, and `scenes.Shot.prompt()`
   names the story object explicitly.
2. **`edit` strategy.** The first *character* shot is rendered fresh by `flux-1-schnell` and
   becomes canonical. Later character shots are **edits** of that same frame by `flux-2-klein`
   with a style+identity anchor. Edits never chain off the previous edit, and environment shots
   stay independent, so drift cannot accumulate.
3. **`sheet` fallback.** If the edit model is unavailable (paid-plan gating, quota, outage), the
   run renders one character sheet and crops per-beat panels — identity is literally identical
   pixels. Chosen automatically, and recorded in `state.json` under `daily.strategy`.

`IMAGE_STRATEGY=edit|sheet|fresh` pins the choice; `--strategy` overrides one run.

### Runtime is emergent, not fixed

Narration is synthesised at **natural pace** (speed 1.0). Speed is only adjusted when the total
falls outside `MIN_SECONDS`–`MAX_SECONDS` (default 30–68 s), so a 6-beat script typically lands
~33–45 s without the drawl of a forced 45 s. Tune with `MIN_SECONDS` / `MAX_SECONDS`.

### Multi-part stories

A story that does not fit one short can span up to **4 parts** (`SERIES_PARTS=1..4`). The arc is
generated once, stored in `state.json`, and each daily run uploads the next part; the queue only
advances after a **successful upload**, so a failed day retries the same part. Parts are titled
`… — Part n/N`. A local-template fallback story is never persisted, so a degraded day cannot
lock placeholder text into the series.

### Zero-spend policy
- No provider assumed to offer a permanent free quota.
- Router rejects providers that require payment.
- Stops on free-limit exhaustion (fail-closed): `429` from Workers AI aborts image generation
  immediately instead of retrying, and the run exits without uploading or advancing the queue.
- Max 8 image attempts per day, max 1 upload per day (`MAX_IMAGE_ATTEMPTS` overrides for dev only).

> Free Workers AI is **10,000 neurons/day per account**. `flux-2-klein` edits bill more per image
> than `flux-1-schnell`, so a 6-beat `edit` run is the practical ceiling per day. That is exactly
> the signal to switch `IMAGE_STRATEGY=sheet` (one render, six crops).

### Local models

Kokoro-82M runs offline; weights are gitignored and provisioned with:

```bash
python audio.py --download      # 311 MB ONNX model + ~4 MB voice bundle into models/kokoro/
python audio.py --say "test line"   # smoke test -> test.wav
```

Available voices in the bundle: `af_heart` (default), `af_bella`, `af_sky`, `am_michael`,
`am_fenrir`, `bm_george`, `af_nicole` — pick with `KOKORO_VOICE=am_michael`.

## Prerequisites

- GitHub repo with secrets configured
- Cloudflare account (Workers AI)
- Google Cloud project (YouTube API OAuth credentials)
- FFmpeg installed on renderer runner (provided in GitHub Actions)

## YouTube channel setup (one hour, once)

**1. Google Cloud project** — <https://console.cloud.google.com> → new project (e.g. `shorts-factory`).

**2. Enable the API** — *APIs & Services → Library → YouTube Data API v3 → Enable*.

**3. OAuth consent screen** — User type **External**; add your own Google account as a test user;
fill the app name and support email. Then **Publish app**. Leaving it in *Testing* is the trap that
kills unattended runs: **refresh tokens expire after 7 days** while an app is in testing status.
The unverified-app warning screen is expected for personal use — click *Advanced → Go to app*.

To move from *Testing* to *In production* Google requires a **homepage URL and a privacy policy
URL** on an **authorised domain** (the *Publish app* button stays disabled until they exist).
`docs/index.html`, `docs/privacy.html` and `docs/terms.html` are ready for GitHub Pages:

1. Push the repo, then *Settings → Pages → Deploy from a branch → `main` / `/docs`*.
2. Branding: Application home page `https://<user>.github.io/<repo>/`, privacy policy
   `https://<user>.github.io/<repo>/privacy.html`.
3. Authorised domains: `<user>.github.io` (bare domain, no scheme or path) → **Save**.

**4. OAuth client** — *Credentials → Create credentials → OAuth client ID →*
**Desktop app** (not Web application; the helper uses a `127.0.0.1` loopback redirect).

**5. Mint the refresh token** — put the two client values into `.env`, then:

```bash
python tools/youtube_auth.py
```

It opens the consent screen, captures the code on loopback, verifies the token against your channel,
and writes `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` / `YOUTUBE_REFRESH_TOKEN` into `.env`
without printing the token. Create the channel first at <https://www.youtube.com/create_channel>
if the script reports none.

**6. Confirm** — `python verify.py --skip-images`, then `python orchestrator.py --render-only`
for a real 1080×1920 file before spending an upload.

### Quota and the audit trap

- Upload costs **1600 units** of the project's 10,000/day; `videos.update` (privacy flip) ~50.
  One short per day is nowhere near the limit — 6/day would still fit.
- **Videos uploaded by a project YouTube hasn't audited can be locked to `private`.** The pipeline
already uploads `private` first (see `publish.upload_video`) so nothing is published unvetted, but if
`publish.set_privacy()` fails to make a video public, that is this restriction, not a bug — it needs
the YouTube API Services audit, or manual publishing from YouTube Studio.
- Verify once by uploading a single test video, flipping it public, then deleting it before the
first scheduled run.

## Secrets required

| Secret | Purpose |
|--------|---------|
| `CFAI_TOKEN` | Cloudflare AI API token |
| `CF_ACCOUNT_ID` | Cloudflare account ID (for Workers AI; falls back to `R2_ACCOUNT_ID`) |
| `YOUTUBE_CLIENT_ID` / `CLIENT_SECRET` / `REFRESH_TOKEN` | YouTube OAuth |
| `OPENROUTER_API_KEY` | Optional — only used if Cloudflare story generation fails |
| `KOKORO_VOICE` | Optional — narration voice (default `af_heart`) |

Local `.env` (gitignored) is read by `config.load_config()`; the same names work as GitHub
Actions secrets.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python audio.py --download          # one-off Kokoro weights

python orchestrator.py --render-only   # stop before upload
python orchestrator.py                 # full run
```

Useful flags: `--offline-images` (test-pattern frames, no quota), `--reuse-frames`
(re-render `build/frames/`, no quota), `--no-audio` (silent cut for debugging),
`--strategy edit|sheet|fresh` (pin the image strategy for one run).

Environment knobs: `IMAGE_STRATEGY`, `MIN_SECONDS`, `MAX_SECONDS`, `SERIES_PARTS`,
`SHOTS_PER_PART`, `PREFERRED_SECONDS`, `KOKORO_VOICE`, `MAX_IMAGE_ATTEMPTS`, `OPENROUTER_MODELS`.

> **Story fallback ordering.** Cloudflare `llama-3.3-70b-fast` → OpenRouter → local template.
> `openrouter/free` is an auto-router that can land on a *reasoning* model which burns the whole
> token budget thinking and returns `content: null`, so the OpenRouter step walks a model list
> (`OPENROUTER_MODELS`, comma separated) and skips empty or rate-limited replies. Upstream free
> models are frequently `429`; when every provider fails the run uses template narration — and
> because a template story is never persisted, the next day retries the real providers.

## Verify against the live providers

```bash
python verify.py                  # beats -> images -> Kokoro -> render, asserts each stage
python verify.py --skip-images    # reuse cached frames, spends no image quota
python verify.py --strategy sheet # check the fallback strategy specifically
```

## Run tests

```bash
python -m pytest tests/ -v
```

## Manual override

Trigger a manual run from the GitHub Actions tab or add `workflow_dispatch`.
