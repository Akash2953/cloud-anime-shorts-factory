# Cloud Anime Shorts Factory

Zero-cost cloud pipeline that autonomously generates and publishes one anime-style YouTube Short per day — no Windows PC needed.

## Architecture

```
GitHub Actions (cron 12:00 UTC) — one upload per day
    └── Python Orchestrator
         ├── Scene plan     Antigravity OAuth → OpenRouter free models → local template
         │                  (Workers AI stays reserved for images)
         │                  shot list: size/camera/place/action/object/role/character/narration
         │                  1 part per short, or up to 7 parts queued across days
         ├── Images         Antigravity OAuth first, Workers AI on any refusal; strategy `edit`:
         │                  first character shot = canon,
         │                  later character shots = edits of it; env/insert shots render fresh
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

55 s → **8 shots ≈ 6.8 s per cut** (`SECONDS_PER_CUT`, clamped to 6–10 shots). Fewer, longer cuts
exist so the voice-over can carry two or three connected sentences per shot: at a 3.4 s rhythm
each beat only fits ~8 words, and thirteen 8-word lines render as thirteen captions rather than a
story. Narration is budgeted from the narrator's measured rate (158 wpm → ~145 words for 55 s) and
a plan whose lines average under 11 words triggers exactly one repair request. Prompts put
**scene first** and the character signature last — a leading character paragraph collapses every
frame into the same portrait (measured: 7 medium portraits vs. 13 varied shots).

### The opening hook (why a new story does not start on weather)

Every plan states a `Hook:` line above the title, and Shot 1's Narration must begin with that
sentence word for word. The hook is one spoken line, 10–16 words, whose job is to make a stranger
who has never seen the story need the next ten seconds: a person, in this story's place, against
one broken rule or one price — a flat fact, not a greeting, a question or a skyline.

This is checked, not hoped for: `parse_hook()` reads the line, `_hook_problem()` rejects a hook
that opens with setup ("once upon a time", "in a world", "long ago", "have you ever") or that is
too short or too long to be spoken, and `_hook_gaps()` compares the stated hook against the first
words Shot 1 actually says. Any failure is a repair instruction, so the planner rewrites the whole
storyboard once instead of shipping the mismatch (`tests/test_scenes.py::TestOpeningHook`,
`tests/test_plan_repair.py`). A past short is the reason this exists: its first line was
"The mist clung to Oakhaven's rooftops when Kael arrived" — atmosphere with nothing at stake.

What the position in the arc changes:

| Position | The opening line must |
|---|---|
| part 1, or a single-part short | state the story's question, in the story's own world |
| a middle or final part (2…N) | **re-hook, not recap**: one clause of load-bearing context (who, what they are chasing, what it now costs), then the new turn. No "previously on" |

The research behind that split: viewers decide inside the first 1–3 s, before any context lands;
curiosity comes from a gap the viewer expects to be closed soon (Loewenstein's information-gap
theory); scenario openings beat question openings by 3–6× in measured short-form data, while
question hooks underperform every other family above 1,000 followers; openings built on setup,
weather or a logo lose the scroll; and serialized "previously on" recaps measurably *lower*
retention versus opening on the new stake. A "This is the story of…" frame is allowed only when
the clause after it already carries the contradiction or the cost ("…a boy who sold memories and
forgot his own sister"), never a bare name — the phrase is a promise, and a promise needs stakes.
The story title is never spoken: it is the upload title, the thumbnail and the channel grid.

Later parts also inherit the narrator: `story_so_far` pins the person and tense, which is how the
part-2 voice flip (third person → "my palm") is meant to stop.

### Place changes

When a shot's `Place` differs from the one before it, the **first clause** of that shot's narration
carries the move ("By the time he reached the Spire…", "Back at the shrine…"). It is fused into a
sentence that already does story work, never a standalone "Later, at the temple." — a whole line
of signposting costs six words of runtime and earns none of them back, while the fused clause
costs nothing and keeps a viewer who looked away for two seconds oriented. The rule lives in the
planner prompt only; it is a taste call, not something the pipeline re-asks over.

### Character consistency (why the pictures hold together)

1. **Art direction first.** Identity never depends on the face: each arc gets its own short
   signature from `scenes.make_character(series_id)` (a costume accent, not a face), pinned in
   `state.json` under `series.character` so the whole arc stays consistent while different arcs
   draw different protagonists. Most cuts are places, objects and backs, and
   `scenes.Shot.prompt()` names the story object explicitly.
2. **`edit` strategy.** The first *character* shot is rendered fresh (Antigravity, or `flux-1-schnell`
   on the Workers AI fallback) and becomes canonical. Later character shots are **edits** of that
   same frame — Antigravity image editing, or `flux-2-klein` when Antigravity refuses — with a style+identity
   anchor. Edits never chain off the previous edit, and environment shots stay independent, so
   drift cannot accumulate.
3. **`sheet` fallback.** If the edit model is unavailable (paid-plan gating, quota, outage), the
   run renders one character sheet and crops per-beat panels — identity is literally identical
   pixels. Chosen automatically, and recorded in `state.json` under `daily.strategy`.

`IMAGE_STRATEGY=edit|sheet|fresh` pins the choice; `--strategy` overrides one run.

### Runtime is emergent, and the narrator never changes pace

Narration is synthesised at **natural pace** (speed 1.0) and is **never time-scaled** — not per
shot, not per part. A long script extends the video instead, because a sped-up read is not
understandable to viewers whose first language is not English, and most of a Shorts feed is
watched that way. Every still is already held for its own line (`beat_track`), so a longer beat
simply holds longer.

Because the word count is the runtime, over-length is handled at plan time, in words: the planner
receives a hard ceiling (`MAX_SECONDS` minus the breaths and the tail hold, converted at 158 wpm)
and a plan above it is re-asked once with the cut named (`_plan_gaps(max_words=…)`). A plan that
still overshoots renders long — with a `WARNING` — until it would stop being a Short: past
`HARD_MAX_SECONDS` (default 175 s) the run refuses to upload rather than speed the narrator up or
publish a non-Short, and the part stays queued.

`MIN_SECONDS` now only sizes the plan (`SHOTS_PER_PART`); it does not slow speech down either.
Tune the slot with `MIN_SECONDS` / `MAX_SECONDS` / `PREFERRED_SECONDS`.

### Subtitles

Word-synced captions need **faster-whisper and Pillow in the same interpreter that runs the
pipeline**. Both are in `requirements.txt`; the GitHub runner installs them, but a local run under
a different Python will not have them:

```bash
python -c "from captions import missing_caption_dependencies; print(missing_caption_dependencies())"
# [] = word-synced captions; ['faster_whisper'] = one caption per beat; ['PIL'] = no captions
```

Degradation is deliberate and announced at startup (`caption_preflight`), because a silent one
once published a short with no subtitles at all:

| Missing | Result |
|---|---|
| nothing | word-synced 1–3 word groups (faster-whisper alignment) |
| `faster_whisper` | **line-level captions** — one strip per beat, timed to the beat holds (`WARNING` at startup) |
| `PIL` | no captions can be rendered (`ERROR` at startup, before image quota is spent) |

### Multi-part stories

A story that does not fit one short can span up to **7 parts**. Set `SERIES_PARTS` to an int
(`1..7`) for a fixed length, or to **`auto`** and let the planner decide per story: on a new
story it states `Arc: N parts` in the same reply that plans part 1, N is clamped to
`PARTS_MIN..PARTS_MAX` (default `1..7`), and the length is then **pinned** in `state.json` so a
later `SERIES_PARTS` change cannot split an arc that is already half-published. A tight idea
comes back as 1 part, a hunt-descent-reunion as 3, a saga as 5–7 — one request, no extra call.

The arc is generated one part per day, appended to `state.json`, and each daily run uploads the
next part; the queue only advances after a **successful upload**, so a failed day retries the same
part. Parts are titled `… — Part n/N`. When the last part is uploaded the next run starts a new
story (and, under `auto`, picks a new length).

Each part is planned knowing its position: parts `1..N-1` are told *not* to resolve and must end
on a hook (`build_tension` / `deliver_payload`), only part `N` may close on a `resolution`. The
planner also receives the narration already told (*story so far*) so a later part continues the
same story instead of inventing a new one. `validate_story_position()` logs a warning whenever a
plan contradicts its position — a middle part that resolves, or a final part that does not — which
is how an open-ended short used to reach the upload step unnoticed. Image prompts are stored on
each shot (`image_generation_prompt`) rather than in a part-indexed dict, so part 2 cannot pick up
part 1's prompt. A local-template fallback story is never persisted, so a degraded day cannot
lock placeholder text into the series.

### Provider priority

**Antigravity OAuth is first** for story planning and image generation when configured. It can use
a manually supplied access token, or refresh access tokens automatically from a refresh token.
Both methods require `ANTIGRAVITY_PROJECT_ID`. Gemini API keys are not used by the pipeline.
An Antigravity refusal (model unavailable, quota, safety block, timeout) falls through to the
next provider:

| Job | 1st | 2nd | 3rd |
|---|---|---|---|
| Story text | Antigravity OAuth (`gemini.py`) | OpenRouter free models | local template |
| Images | Antigravity OAuth | Workers AI `flux-1-schnell` / `flux-2-klein` | — |

Antigravity model names are overridable (`ANTIGRAVITY_TEXT_MODEL`,
`ANTIGRAVITY_IMAGE_MODEL`). An Antigravity `429` falls back to OpenRouter for story text or
Workers AI for images. Antigravity image requests specify a `9:16` aspect ratio, and returned
images are center-cropped to that exact ratio before use. `images.fresh_image` / `images.edit_image`
choose the image provider, so the strategy engine (`edit` / `sheet` / `fresh`) behaves identically
on either provider.

### Zero-spend policy
- No provider assumed to offer a permanent free quota.
- Router rejects providers that require payment.
- Antigravity is first priority but never assumed available: an expired OAuth token costs one
  failed request, then the free Workers AI path renders the image.
- Story planning never touches Workers AI: its daily neurons belong to image generation, so
  planning goes Antigravity → OpenRouter → local template (`plan_shots`/`plan_arc`/
  `plan_shorts_detailed` accept no Cloudflare credentials at all).
- Stops on free-limit exhaustion (fail-closed): `4006` / `429` from Workers AI aborts image
  generation immediately instead of retrying, and the run exits without uploading or advancing
  the queue. The error code is logged, so "quota spent" is never mistaken for "rate limited".
- Max 16 image attempts per day (one per beat for a 13-shot short, plus headroom for retries),
  max 1 upload per day (`MAX_IMAGE_ATTEMPTS` overrides for dev only).

**What 10,000 neurons actually buys** (rates from
<https://developers.cloudflare.com/workers-ai/platform/pricing/>, `$0.011 / 1,000 neurons`,
reset at 00:00 UTC). Images are 1024×1024, i.e. 4 tiles of 512×512:

| call | billing | neurons | per 10k/day |
|---|---|---|---|
| `flux-1-schnell` fresh | 4 tiles × 4.80 + 4 steps × 9.60 | **~58** | **~170 images** |
| `flux-2-klein-4b` edit | 4 input × 5.37 + 4 output × 26.05 | **~126** | **~80 edits** |
| `sheet` (one render, N crops) | one schnell image | ~58 | ~170 sheets |

So a 13-shot short costs **~750 neurons** (`fresh`, 13 images) or **~1,450 neurons** (`edit`:
1 canonical + 10 edits + 2 environment renders) — about **7–13 shorts per day** inside the free
allowance. The real ceilings are one upload/day and OpenRouter's 50 free planner requests/day
(1–2 per short), not the neuron budget. With Antigravity OAuth configured these numbers are the
*fallback* budget: neurons are only spent on days Antigravity refuses. Keep `IMAGE_STRATEGY=sheet` for the case that *is* tight:
the planner's free models are down, or a run has to be re-rendered several times in one day.

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
- Antigravity OAuth credentials and its Google project ID (optional; the free fallbacks below work
  without them)
- Cloudflare account (Workers AI image fallback)
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
| `ANTIGRAVITY_REFRESH_TOKEN` | Optional first-priority provider for story text and images; access tokens refresh automatically |
| `ANTIGRAVITY_OAUTH_CLIENT_ID` / `ANTIGRAVITY_OAUTH_CLIENT_SECRET` | OAuth client credentials used by the local sign-in and token refresh; secret is optional for desktop clients |
| `ANTIGRAVITY_ACCESS_TOKEN` | Optional short-lived token for manual/local use; it does not refresh automatically |
| `ANTIGRAVITY_PROJECT_ID` | Required with Antigravity; set as a GitHub Actions variable |
| `CFAI_TOKEN` | Cloudflare AI API token (image fallback) |
| `CF_ACCOUNT_ID` | Cloudflare account ID (for Workers AI; falls back to `R2_ACCOUNT_ID`) |
| `YOUTUBE_CLIENT_ID` / `CLIENT_SECRET` / `REFRESH_TOKEN` | YouTube OAuth |
| `OPENROUTER_API_KEY` | Story planning fallback when Antigravity refuses |
| `KOKORO_VOICE` | Optional — narration voice (default `af_heart`) |

Local `.env` (gitignored) is read by `config.load_config()`; the same names work as GitHub
Actions secrets or variables. The Antigravity token is sent only in the Authorization header and
is never printed by `verify.py`.

### Antigravity OAuth sign-in

Set `ANTIGRAVITY_OAUTH_CLIENT_ID` (and, if your OAuth client requires it,
`ANTIGRAVITY_OAUTH_CLIENT_SECRET`) in `.env`, then run:

```bash
python tools/antigravity_auth.py
```

Approve access in the browser. The helper uses a loopback callback and PKCE, discovers the
Antigravity project when available, and saves the refresh token and project ID to the ignored
`.env` file without printing token values. For GitHub Actions, copy `ANTIGRAVITY_REFRESH_TOKEN`
and the OAuth client values into repository Secrets, and set `ANTIGRAVITY_PROJECT_ID` as a
repository Variable. Do not put an access token in Actions; it expires and cannot refresh itself.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python audio.py --download          # one-off Kokoro weights

python orchestrator.py --render-only   # stop before upload
python orchestrator.py                 # full run
```

Useful flags: `--render-only` (stop before upload), `--plan-only` (story plan only),
`--reuse-frames` (re-render `build/frames/`, no image quota), `--part N` (plan/preview part N
of the arc without advancing the publish queue). Image strategy is set with `IMAGE_STRATEGY`.

Environment knobs: `IMAGE_STRATEGY`, `MIN_SECONDS`, `MAX_SECONDS`, `SERIES_PARTS` (`1..7` or
`auto`), `PARTS_MIN`, `PARTS_MAX`, `SHOTS_PER_PART`, `PREFERRED_SECONDS`, `KOKORO_VOICE`,
`MAX_IMAGE_ATTEMPTS`, `ANTIGRAVITY_REFRESH_TOKEN`, `ANTIGRAVITY_OAUTH_CLIENT_ID`,
`ANTIGRAVITY_OAUTH_CLIENT_SECRET`,
`ANTIGRAVITY_ACCESS_TOKEN`, `ANTIGRAVITY_PROJECT_ID`, `ANTIGRAVITY_TEXT_MODEL`,
`ANTIGRAVITY_IMAGE_MODEL`, `OPENROUTER_MODELS`.

> **Story fallback ordering.** Antigravity OAuth (`ANTIGRAVITY_TEXT_MODEL`, default
> `gemini-3-flash`) → OpenRouter (`OPENROUTER_MODELS`, comma separated; defaults walk
> nemotron-3-ultra → north-mini-code → gemma-4-26b → `openrouter/free`) → local template.
> Free reasoning models can spend the entire completion budget thinking and return `content: null`
> (measured: `reasoning_tokens: 2169` of 1999, content empty), so planning runs with
> `max_tokens=4096` and logs `finish_reason` / `reasoning_tokens` on every reply; a truncated
> storyboard (`finish_reason=length`) is named in the log instead of quietly coming back short.
> Upstream free models are also frequently `429`. When every model fails the run uses template
> narration — and because a template story is never persisted, the next day retries real ones.

## Verify against the live providers

```bash
python verify.py                  # beats -> images -> Kokoro -> render, asserts each stage
python verify.py --skip-images    # reuse cached frames, spends no image quota
python verify.py --strategy sheet # check the fallback strategy specifically
```

The header prints which providers are configured (presence only), including the selected
Antigravity text/image models and the image fallback chain
`Antigravity -> flux-1-schnell + flux-2-klein`. Antigravity defaults to `gemini-3-flash` for text
and `gemini-3.1-flash-image` for images.

## Run tests

```bash
python -m pytest tests/ -v
```

## Manual override

Trigger a manual run from the GitHub Actions tab or add `workflow_dispatch`.
