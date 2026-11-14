"""Word-synced captions for the narrated short.

Pipeline slot: called once, after Kokoro narration has been written to WAV
and per-beat hold durations are known, before render_narration burns
anything.

Stages:
    1. align_words        faster-whisper -> list of WordCue with ms-precise timing
    2. pack_groups        1/2/3-word visible windows, adaptive to VO gap timing
    3. classify_word       stopword-vs-content -> color slot for each word
    4. render_group_pngs   one 1080-wide PNG per group, ready for ffmpeg overlay

Graceful degrade: when word alignment fails, `build_caption_overlays` falls back to
LINE-LEVEL captions derived from the narration text and the beat timing, so the short always
ships with subtitles. A missing faster-whisper install must never mean a silent video.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# Defaults — match the spec you signed off on (350 ms gap = group, 700 ms long word alone).
# Lead-in + tail are kept small so each word group lands on screen tightly around its audio and
# slides out before the next one finishes its lead-in: total per-transition visual duration now
# sits around 140 ms instead of the earlier ~380 ms, which read as a half-second stagger.
DEFAULT_GAP_TIGHT_MS: int = 350
DEFAULT_LONG_WORD_MS: int = 700
DEFAULT_MAX_GROUP: int = 3
DEFAULT_LEAD_IN_MS: int = 40
DEFAULT_TAIL_MS: int = 100
DEFAULT_OVERLAP_MS: int = 20

# Stopwords cover the example case directly: "she ran towards" -> she=stop, ran=accent, towards=stop.
STOPWORDS: frozenset[str] = frozenset({
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "every", "each", "no",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "hers", "its", "our", "their", "mine", "yours", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    # auxiliary / copular / modal verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done",
    "has", "have", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # conjunctions / connectives
    "and", "but", "or", "nor", "so", "yet", "for", "because", "although", "though", "while",
    "if", "unless", "until", "when", "whenever", "where", "wherever", "whether",
    "as", "since", "than", "that", "after", "before", "once",
    # prepositions
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from", "up", "down",
    "out", "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "near", "far", "toward", "towards", "onto", "upon", "within", "without", "across",
    "behind", "beyond", "around", "beside", "among", "amid",
    # common adverbs / fillers we don't want shouting in captions
    "not", "no", "yes", "just", "only", "also", "very", "really", "much", "many",
    "still", "now", "never", "always", "often", "sometimes", "perhaps", "maybe",
    # deictics the eye already forgives
    "here", "there", "where",
})

# Soft-mute bypass: content words that visually carry no story weight. Forces them to white
# so a short doesn't end up with "go" or "said" in neon pink.
SOFT_MUTE: frozenset[str] = frozenset({
    "go", "goes", "going", "gone", "went",
    "come", "comes", "coming", "came",
    "get", "gets", "getting", "got", "gotten",
    "look", "looks", "looked", "looking", "see", "saw", "seen",
    "say", "says", "said", "saying",
    "ask", "asks", "asked", "asking",
    "tell", "tells", "told", "telling",
    "make", "makes", "made", "making",
    "let", "lets", "letting",
    "put", "puts", "putting",
    "take", "takes", "took", "taken", "taking",
    "give", "gives", "gave", "given", "giving",
    "keep", "keeps", "kept", "keeping",
    "feel", "feels", "felt", "feeling",
    "seem", "seems", "seemed", "seeming",
})

DEFAULT_ACCENT_PALETTE: tuple[str, ...] = ("#FF2D55", "#FFD60A", "#00E5FF")


# Font discovery — Linux CI installs DejaVu-Bold; Windows dev has Arial Bold; macOS dev has
# /Library/Fonts Arial. Picks the first hit; logs and falls back to Pillow's bitmap default
# (which is ugly but never crashes).
DEFAULT_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\DejaVuSans-Bold.ttf",
)


@dataclass(frozen=True)
class WordCue:
    """One transcribed word, with timing relative to the joined narration WAV."""
    start: float   # seconds
    end: float     # seconds
    text: str      # raw lowercase token, may carry leading/trailing punctuation

    @property
    def duration_ms(self) -> int:
        return int(round((self.end - self.start) * 1000))


@dataclass(frozen=True)
class GroupCue:
    """A visible 1- to 3-word window that overlays ONE PNG during ffmpeg rendering."""
    start: float   # show window start (s) — typically first_word.start - lead_in
    end: float     # show window end (s)   — typically last_word.end + tail
    words: tuple[WordCue, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# ----------------------------------------------------------------------------------------------
# Stage 1: alignment
# ----------------------------------------------------------------------------------------------


def missing_caption_dependencies() -> list[str]:
    """Importable modules the caption stage needs, which the plain pipeline does not.

    `faster_whisper` missing only downgrades word sync to line-level captions; Pillow missing
    means no caption PNGs can be rendered at all. The orchestrator checks this before spending
    image quota, because a caption-less publish used to look like a healthy run.
    """
    missing: list[str] = []
    for module in ("faster_whisper", "PIL"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def align_words(wav_path: Path, narration_text: str, *, model_size: str = "base.en",
                language: str = "en") -> list[WordCue]:
    """Run faster-whisper on a single WAV and return one WordCue per token.

    `narration_text` is fed as `initial_prompt`. Whisper hallucinates wildly without it; with
    the exact narration pasted in, we get drift near zero on narration audio.

    Hidden behind try/except because the alignment is best-effort: if faster-whisper isn't
    installed or the model can't be loaded (offline dev), the call returns [] and the caller
    falls back to line-level captions.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        # WARNING, not INFO: the caller uses this to decide whether to fall back to line-level
        # captions. An INFO line here is how a published short shipped with no subtitles at all.
        logger.warning("faster-whisper not installed (%s); word alignment unavailable — "
                       "falling back to line-level captions", exc)
        return []

    if not wav_path.exists():
        logger.warning("align_words: %s does not exist", wav_path)
        return []

    try:
        # CPU int8 keeps the install self-contained — no GPU runtime dependency.
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        cues = _transcribe_words(model, wav_path, narration_text, language)
        cues = _repair_gaps(model, wav_path, cues, language)
        cues = _prune_hallucinations(cues, wav_path)
        cues = _trim_trailing_hallucination(cues, narration_text)
        if not cues:
            logger.warning("faster-whisper returned no word timings for %s", wav_path)
        return cues
    except Exception as exc:  # noqa: BLE001 — fail loudly in logs, not in pipeline
        logger.warning("align_words failed (%s); falling back to line-level captions", exc)
        return []


def _transcribe_words(model, wav_path: Path, initial_prompt: str | None,
                      language: str) -> list[WordCue]:
    """One decode pass -> WordCues. Shared by the main pass and the gap repairer.

    `condition_on_previous_text=False` is load-bearing. With it left at the default (True) and
    an `initial_prompt` supplied, whisper's decoder can skip a whole window: on the Inari short
    it emitted a segment ending at 27.44 s and then the next one starting at 33.92 s, silently
    dropping "where she last slept. The house holding its breath..." — which is exactly the
    four seconds of video that rendered with no captions. Turning conditioning off made the
    same audio produce 165 words with no gaps.
    """
    segments, _info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
        # Passing the full narration biases decode toward the literal text. Without this,
        # whisper "improves" Kokoro's pronunciation into something plausible but moved
        # ("Inari Station" came back as "ARRI station").
        initial_prompt=initial_prompt or None,
    )

    cues: list[WordCue] = []
    for seg in segments:
        for w in (seg.words or []):
            # faster-whisper sometimes emits None for start/end at segment edges.
            if w.start is None or w.end is None:
                continue
            token = (w.word or "").strip()
            if not token:
                continue
            # whisper wraps tokens with leading space: " she" -> "she"
            cues.append(WordCue(start=float(w.start), end=float(w.end), text=token.lower()))
    return cues


def is_hallucination_token(text: str) -> bool:
    """True for tokens whisper invents on silence: bare digits, symbols, stray punctuation.

    The Inari short ended with `8. 7 8. 8.` over the 0.6 s of `apad` silence — whisper's
    well-known silence hallucination. Every real narration word contains at least one letter,
    so a token with no alphabetic character is never something Kokoro spoke.
    """
    return not any(ch.isalpha() for ch in text)


def _prune_hallucinations(words: list[WordCue], wav_path: Path,
                          *, min_rms: float = 0.002) -> list[WordCue]:
    """Drop invented tokens: no letters, or sitting in digital silence.

    Two independent guards because whisper hallucinates in two shapes — digits/symbols on
    silence, and whole phrases ("Thank you.") on silence. The letter check catches the first;
    the RMS check catches both, since the tail of the track is `apad` (exact zeros).
    """
    if not words:
        return words

    kept = [w for w in words if not is_hallucination_token(w.text)]
    dropped = len(words) - len(kept)

    try:
        import numpy as np
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        silent: list[WordCue] = []
        for w in kept:
            lo = max(0, int(w.start * sr))
            hi = min(len(audio), int(w.end * sr))
            if hi <= lo:
                silent.append(w)
                continue
            rms = float(np.sqrt(np.mean(np.square(audio[lo:hi]))))
            if rms < min_rms:
                silent.append(w)
        if silent:
            kept = [w for w in kept if w not in silent]
            dropped += len(silent)
    except Exception as exc:  # noqa: BLE001 — pruning is best-effort
        logger.debug("silence pruning skipped (%s)", exc)

    if dropped:
        logger.info("dropped %d hallucinated caption token(s)", dropped)
    return kept


def _normalise_token(text: str) -> str:
    """Lowercase, strip surrounding punctuation — the form used for vocabulary matching."""
    return text.strip(".,;:!?\"'()[]{}<>\u2014\u2013-").lower()


def _trim_trailing_hallucination(words: list[WordCue], narration_text: str,
                                 *, max_trim: int = 6) -> list[WordCue]:
    """Drop invented tokens at the end of the track by checking them against the script.

    Whisper hallucinates on the decaying tail of the final word, where there is still real
    energy — so an RMS test cannot see it. On the Inari short it appended `immediate aid.`
    after `stayed.` (and `8. 7 8. 8.` on an earlier decode). We know the exact narration, so
    any trailing token that does not appear in the script is invented.

    Only the tail is trimmed: a mis-transcribed word in the middle sits at a real timestamp and
    is still the right caption, so it is left alone. `max_trim` bounds the damage if the final
    word itself was mis-heard.
    """
    if not words or not narration_text.strip():
        return words

    vocab = {_normalise_token(t) for t in narration_text.split()}
    vocab.discard("")
    if not vocab:
        return words

    end = len(words)
    while end > 0 and _normalise_token(words[end - 1].text) not in vocab:
        end -= 1
    trimmed = len(words) - end
    if trimmed == 0:
        return words
    if trimmed > max_trim:
        logger.warning("refusing to trim %d trailing tokens (max %d); keeping transcript",
                       trimmed, max_trim)
        return words
    logger.info("trimmed %d trailing hallucinated token(s): %s", trimmed,
                " ".join(w.text for w in words[end:]))
    return words[:end]


def find_gaps(words: list[WordCue], *, max_gap_s: float = 1.2) -> list[tuple[float, float]]:
    """Spans with no word coverage longer than `max_gap_s`.

    Pure helper so the repair logic is testable without a whisper model. Beat gaps in the
    narration are 0.35 s, so 1.2 s means a genuinely unaligned stretch of speech.
    """
    gaps: list[tuple[float, float]] = []
    for a, b in zip(words, words[1:]):
        if b.start - a.end > max_gap_s:
            gaps.append((a.end, b.start))
    return gaps


def _repair_gaps(model, wav_path: Path, words: list[WordCue], language: str,
                 *, max_gap_s: float = 1.2, min_rms: float = 0.004) -> list[WordCue]:
    """Re-decode only the unaligned spans and splice the recovered words back in.

    Safety net for the decoder-skip class of failure: if whisper still drops a window, the
    captions would silently vanish for that stretch of video. Slicing the audio to the gap and
    decoding it fresh (no prior context, no initial_prompt) recovers the words in practice, and
    the timeline stays anchored because we re-add the slice offset.

    Gaps that are actually silence are skipped via an RMS check, so a planned silent beat is
    not turned into invented text.
    """
    gaps = find_gaps(words, max_gap_s=max_gap_s)
    if not gaps:
        return words

    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        return words

    recovered: list[WordCue] = []
    for start, end in gaps:
        slice_words = _decode_span(model, wav_path, start, end, language, min_rms=min_rms)
        if slice_words:
            recovered.extend(slice_words)
            logger.info("caption gap %.2fs-%.2fs repaired with %d word(s)",
                        start, end, len(slice_words))

    if not recovered:
        return words
    merged = sorted(words + recovered, key=lambda w: w.start)
    # A repair pass can overlap a neighbouring word by a frame; drop exact duplicates only.
    deduped: list[WordCue] = []
    for w in merged:
        if deduped and abs(w.start - deduped[-1].start) < 0.02 and w.text == deduped[-1].text:
            continue
        deduped.append(w)
    return deduped


def _decode_span(model, wav_path: Path, start: float, end: float, language: str,
                 *, min_rms: float) -> list[WordCue]:
    """Decode one gap slice; return [] when the span is silence."""
    import tempfile
    import numpy as np
    import soundfile as sf  # type: ignore

    audio, sr = sf.read(str(wav_path), dtype="float32")
    lo = max(0, int(start * sr))
    hi = min(len(audio), int(end * sr))
    if hi <= lo:
        return []
    chunk = audio[lo:hi]
    rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
    if rms < min_rms:
        logger.debug("caption gap %.2fs-%.2fs is silent (rms %.4f); not repairing",
                     start, end, rms)
        return []

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        sf.write(str(tmp), chunk, sr, subtype="PCM_16")
        local = _transcribe_words(model, tmp, None, language)
    finally:
        tmp.unlink(missing_ok=True)

    # Re-anchor to the full track.
    return [WordCue(start=w.start + start, end=w.end + start, text=w.text) for w in local]


def filter_usable_words(words: Iterable[WordCue], audio_duration: float,
                        *, drift_tol: float = 0.12) -> list[WordCue]:
    """Drop a word transcript whose total span drifts too far from the actual WAV.

    Whisper occasionally re-times the whole clip ~10% faster or slower than the WAV when the
    language model vetoes the initial_prompt. If that drift exceeds `drift_tol`, captions
    would visibly fall behind / run ahead — skip word mode for this short.
    """
    words = list(words)
    if not words or audio_duration <= 0:
        return []
    span = words[-1].end - words[0].start
    rel = abs(span - audio_duration) / audio_duration
    if rel > drift_tol:
        logger.warning("alignment drift %.0f%% exceeds %d%% tolerance; dropping word captions",
                       rel * 100, int(drift_tol * 100))
        return []
    return words


# ----------------------------------------------------------------------------------------------
# Stage 2: grouping
# ----------------------------------------------------------------------------------------------


def pack_groups(
    words: list[WordCue],
    *,
    gap_tight_ms: int = DEFAULT_GAP_TIGHT_MS,
    long_word_ms: int = DEFAULT_LONG_WORD_MS,
    max_group: int = DEFAULT_MAX_GROUP,
    lead_in_ms: int = DEFAULT_LEAD_IN_MS,
    tail_ms: int = DEFAULT_TAIL_MS,
    overlap_ms: int = DEFAULT_OVERLAP_MS,
) -> list[GroupCue]:
    """Greedy 1/2/3-word groups driven by the actual VO gaps.

    "she ran towards" -> [["she","ran"], ["towards"]] at 350 ms.
    A dramatic single word like "*never.*" (≥0.7 s) defaults to a 1-word group, even if
    its neighbours would have grouped with it.
    """
    if not words:
        return []

    def _window(sub: list[WordCue]) -> tuple[float, float]:
        first, last = sub[0], sub[-1]
        # Small lead-in so the eye settles on the word before its sound; generous tail so the
        # next group can come in without a hard cut. First group also clamps at 0 so subtitles
        # never flash before t=0 in the joined track.
        start = max(0.0, first.start - lead_in_ms / 1000.0)
        end = last.end + tail_ms / 1000.0
        if groups:
            # Allow the next group in slightly before this one fully exits.
            start = max(start, groups[-1].end - overlap_ms / 1000.0)
        return start, end

    groups: list[GroupCue] = []
    i = 0
    while i < len(words):
        # Long word? Solo group, advance.
        if words[i].duration_ms >= long_word_ms:
            start, end = _window([words[i]])
            groups.append(GroupCue(start=start, end=end, words=(words[i],)))
            i += 1
            continue

        # Pack up to max_group tight-clustered words; never include a long word here — the
        # outer loop will give it a solo group instead. Stop extending the same group when the
        # next word's start-time gap exceeds the rhythm threshold OR we've hit max_group.
        cur: list[WordCue] = [words[i]]
        j = i + 1
        while j < len(words) and len(cur) < max_group:
            gap_ms = int(round((words[j].start - words[j - 1].start) * 1000))
            if gap_ms > gap_tight_ms:
                break
            if words[j].duration_ms >= long_word_ms:
                # Skip rather than pack: the next iteration handles the long word alone.
                break
            cur.append(words[j])
            j += 1

        start, end = _window(cur)
        groups.append(GroupCue(start=start, end=end, words=tuple(cur)))
        i = j

    return groups


# ----------------------------------------------------------------------------------------------
# Stage 3: classification
# ----------------------------------------------------------------------------------------------


def classify_word(text: str, *, palette: list[str]) -> tuple[str, int]:
    """Return ('white', -1) for stopwords and SOFT_MUTE tokens, ('accent', i) otherwise.

    `i` is the position in `palette` (cyclic by occurrence at call site).
    """
    bare = text.strip(".,;:!?\"'()[]{}<>").lower()
    if bare in STOPWORDS or bare in SOFT_MUTE:
        return ("white", -1)
    return ("accent", 0)  # caller picks the actual index when iterating


# ----------------------------------------------------------------------------------------------
# Stage 4: PNG rendering
# ----------------------------------------------------------------------------------------------


def _resolve_font(font_path: str | None) -> tuple[str, str]:
    """Return (path, source) where source in {'explicit','candidate','builtin','missing'}.

    Logs the pick so misconfigured CI surfaces immediately.
    """
    if font_path and Path(font_path).exists():
        return (font_path, "explicit")
    for cand in DEFAULT_FONT_CANDIDATES:
        if Path(cand).exists():
            return (cand, "candidate")
    return ("", "missing")


def render_group_pngs(
    groups: list[GroupCue],
    out_dir: Path,
    *,
    accent_palette: tuple[str, ...] = DEFAULT_ACCENT_PALETTE,
    font_path: str | None = None,
    canvas_width: int = 1080,
    font_size: int = 130,
    margin_x: int = 80,
) -> list[tuple[Path, float, float]]:
    """Render one transparent RGBA PNG per group, ready for an ffmpeg overlay filter.

    Returns [(png_path, start_seconds, end_seconds), ...] — ready to drop into render_narration.
    Each PNG is 1080-wide and ~250 px tall; positioned in the bottom safe area by render.py.
    """
    if not groups:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear strips from a previous render: a shorter caption list would otherwise leave orphan
    # cap_NN.png files behind, which makes it impossible to tell what the video actually used.
    for stale in out_dir.glob("cap_*.png"):
        stale.unlink(missing_ok=True)

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        logger.info("Pillow not installed; cannot render caption PNGs")
        return []

    chosen, source = _resolve_font(font_path)
    if not chosen:
        logger.warning("no caption font found and Pillow builtin default would be ugly; skipping PNGs")
        return []

    # Load both bold and (if available) regular for shadow. We use bold for everything to keep
    # strokes uniform.
    try:
        base_font = ImageFont.truetype(chosen, font_size)
    except OSError as exc:
        logger.warning("could not load font %s (%s); skipping PNGs", font_path or chosen, exc)
        return []

    results: list[tuple[Path, float, float]] = []

    for idx, group in enumerate(groups):
        # Each group may contain one or two accent words; pick distinct palette slots cyclically.
        token_specs: list[tuple[str, str, str]] = []
        accent_idx = 0
        for w in group.words:
            kind, _ = classify_word(w.text, palette=list(accent_palette))
            if kind == "white":
                token_specs.append((w.text, "#FFFFFF", "white"))
            else:
                color = accent_palette[accent_idx % len(accent_palette)]
                accent_idx += 1
                token_specs.append((w.text, color, "accent"))

        # Auto-shrink if the laid-out text would overflow the canvas.
        font = base_font
        try:
            def _measure(font_):
                widths = []
                for tok, _, _ in token_specs:
                    box = font_.getbbox(tok)
                    widths.append(box[2] - box[0])
                spaces = [int(font_.getlength(" "))] * (len(token_specs) - 1)
                return widths, spaces

            widths, spaces = _measure(font)
            total = sum(widths) + sum(spaces)
            if total > canvas_width - 2 * margin_x and font_size > 70:
                scale = (canvas_width - 2 * margin_x) / total
                shrunk = max(60, int(font_size * scale))
                font = ImageFont.truetype(chosen, shrunk)
                widths, spaces = _measure(font)
                total = sum(widths) + sum(spaces)
        except Exception:  # noqa: BLE001
            # Measure threw (rare odd font). Fall back to unclipped render; ffmpeg has harder
            # center-align, but the layout won't crash.
            widths = [font.getbbox(t)[2] - font.getbbox(t)[0] for t, _, _ in token_specs]
            spaces = [int(font.getlength(" "))] * (len(token_specs) - 1)
            total = sum(widths) + sum(spaces)

        # Compute strip height from the tallest word's metrics.
        try:
            ascent, descent = font.getmetrics()
            strip_h = ascent + descent + 160
        except AttributeError:  # very old Pillow
            strip_h = font_size + 160

        img = Image.new("RGBA", (canvas_width, strip_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        x = (canvas_width - total) / 2
        y_top = (strip_h - (ascent + descent)) / 2 if 'ascent' in dir() else 80

        for i, ((text, color, _kind), tw) in enumerate(zip(token_specs, widths)):
            # Drop shadow (3 stacked passes for soft falloff).
            if 'ascent' in dir():
                shadow_y = y_top
            else:
                shadow_y = 80
            for off in ((0, 5, 150), (0, 3, 130), (1, 1, 110)):
                draw.text((x + off[0], shadow_y + off[1]), text, font=font, fill=(0, 0, 0, off[2]))
            # Black stroke + colored fill.
            draw.text(
                (x, shadow_y), text, font=font, fill=color,
                stroke_width=8, stroke_fill=(0, 0, 0, 255),
            )
            x += tw + (spaces[i] if i < len(spaces) else 0)

        out = out_dir / f"cap_{idx:02d}.png"
        img.save(out, format="PNG")
        results.append((out, group.start, group.end))

    logger.info("rendered %d caption PNGs (%s font=%s)", len(results), source, chosen)
    return results


# ----------------------------------------------------------------------------------------------
# One-stop entrypoint
# ----------------------------------------------------------------------------------------------


def _draw_subscribe_button(draw, x: float, y: float, height: int, font_path: str,
                           *, label: str = "SUBSCRIBE") -> int:
    """Draw a YouTube-style subscribe button and return its width.

    The CTA block shows the button instead of the word "subscribe": a red pill with white caps
    is read as an action at a glance, where the word is read as more narration.
    """
    from PIL import ImageFont  # type: ignore

    text_size = max(18, int(height * 0.42))
    font = ImageFont.truetype(font_path, text_size)
    pad_x = int(height * 0.42)
    text_w = int(draw.textlength(label, font=font))
    width = text_w + pad_x * 2
    radius = height // 2

    # Soft shadow so the pill separates from a bright frame.
    draw.rounded_rectangle([x + 3, y + 5, x + width + 3, y + height + 5],
                           radius=radius, fill=(0, 0, 0, 120))
    draw.rounded_rectangle([x, y, x + width, y + height], radius=radius, fill=(204, 0, 0, 255))
    draw.text((x + pad_x, y + (height - text_size) / 2 - text_size * 0.12), label,
              font=font, fill=(255, 255, 255, 255))
    return width


def render_cta_block(
    text: str,
    out_path: Path,
    *,
    font_path: str | None = None,
    canvas_width: int = 1080,
    font_size: int = 96,
    margin_x: int = 70,
    accent: str = "#FFD60A",
) -> Path | None:
    """Render the whole CTA as one centred block, with a subscribe button in place of the word.

    Unlike the story captions this is not word-synced: the ask is one sentence, and splitting it
    into 2-3 word flashes makes it unreadable. The block is laid out once, centred, and held for
    the whole CTA beat.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        logger.info("Pillow not installed; cannot render the CTA block")
        return None

    chosen, source = _resolve_font(font_path)
    if not chosen:
        logger.warning("no CTA font found; skipping the CTA block")
        return None

    # Tokenise into words, marking the one that becomes a button. A sentence boundary forces a
    # new line, so the button never lands mid-sentence — "...for you! [SUBSCRIBE] to keep them
    # coming!" reads as an ask, where a button in the middle of a clause reads as a typo.
    tokens: list[tuple[str, bool, bool]] = []  # (word, is_button, starts_new_line)
    for sentence in re.split(r"(?<=[!?.])\s+", text.strip()):
        words = sentence.split()
        for k, word in enumerate(words):
            bare = word.strip(".,;:!?\"'()[]{}<>").lower()
            tokens.append((word, bare == "subscribe", k == 0))
    if not tokens:
        return None

    max_width = canvas_width - 2 * margin_x

    def layout(size: int):
        font = ImageFont.truetype(chosen, size)
        # The button is deliberately oversized relative to the text: at 0.78x the font size it
        # read as a footnote, so it is 1.56x (2x the old height) and the line it sits on grows
        # to fit it.
        button_h = int(size * 1.56)
        space = int(font.getlength(" "))
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        lines: list[list[tuple[str, bool, int]]] = [[]]
        used = 0
        for word, is_button, new_line in tokens:
            w = _measure_button(probe, button_h, chosen) if is_button else int(font.getlength(word))
            if new_line and lines[-1]:
                lines.append([])
                used = 0
            need = w if not lines[-1] else used + space + w
            if lines[-1] and need > max_width:
                lines.append([])
                used = 0
                need = w
            lines[-1].append((word, is_button, w))
            used = need
        return font, button_h, space, lines

    size = font_size
    font, button_h, space, lines = layout(size)
    # Shrink while the block is too tall, or while a line is left holding a single orphan word
    # ("...waiting for / you!"), which looks like a mistake at thumbnail size.
    while size > 48 and (len(lines) > 3 or any(len(line) == 1 for line in lines)):
        size -= 6
        font, button_h, space, lines = layout(size)

    ascent, descent = font.getmetrics()
    text_h = ascent + descent
    # A line holding the button is as tall as the button, so the text and the pill can be
    # vertically centred against each other instead of the pill overflowing the line.
    line_heights = [max(int(text_h * 1.22), button_h + 24) if any(b for _, b, _ in line)
                    else int(text_h * 1.22) for line in lines]
    block_h = sum(line_heights) + 40
    img = Image.new("RGBA", (canvas_width, block_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Soft scrim behind the block. At the centre of the frame the CTA sits over whatever the
    # artist drew there — a bright sky or a face — so the text needs its own floor to stay legible.
    draw.rounded_rectangle([36, 2, canvas_width - 36, block_h - 2], radius=28,
                           fill=(0, 0, 0, 135))

    y = 20
    for line, line_h in zip(lines, line_heights):
        total = sum(w for _, _, w in line) + space * (len(line) - 1)
        x = (canvas_width - total) / 2
        text_y = y + (line_h - text_h) / 2
        for word, is_button, w in line:
            if is_button:
                _draw_subscribe_button(draw, x, y + (line_h - button_h) / 2, button_h, chosen)
            else:
                for off in ((0, 5), (0, 3), (1, 1)):
                    draw.text((x + off[0], text_y + off[1]), word, font=font, fill=(0, 0, 0, 150))
                draw.text((x, text_y), word, font=font, fill=accent,
                          stroke_width=max(5, size // 20), stroke_fill=(0, 0, 0, 255))
            x += w + space
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    logger.info("CTA block written: %s (%d line(s), %s font)", out_path, len(lines), source)
    return out_path


def _measure_button(draw, height: int, font_path: str, *, label: str = "SUBSCRIBE") -> int:
    """Width the subscribe button will occupy, so layout can wrap around it."""
    from PIL import ImageFont  # type: ignore

    text_size = max(18, int(height * 0.42))
    font = ImageFont.truetype(font_path, text_size)
    return int(draw.textlength(label, font=font)) + int(height * 0.42) * 2


def build_line_level_cues(
    narration_text: str,
    *,
    segments: list[tuple[str, float, float]] | None = None,
    audio_duration: float = 0.0,
) -> list[GroupCue]:
    """One caption per narration beat, timed against the rendered beat holds.

    This is the fallback used when word alignment is unavailable (faster-whisper not installed,
    model download blocked, alignment returned nothing). It is deliberately simpler than
    word-synced captions: one strip per beat, held for the whole beat.

    `segments` is (text, start_s, end_s) per spoken beat. When it is absent the narration text
    is split into sentence-like lines and spread evenly across `audio_duration`, which still
    yields readable subtitles instead of none.
    """
    if segments:
        groups: list[GroupCue] = []
        for text, start, end in segments:
            words = tuple(WordCue(start=start, end=end, text=word)
                          for word in str(text).split())
            if not words:
                continue
            if audio_duration > 0:
                end = min(end, audio_duration)
                if start >= audio_duration:
                    continue
            groups.append(GroupCue(start=start, end=end, words=words))
        return groups

    # No beat timing: spread the lines across the track. Coarse, but subtitles beat silence.
    lines = [line.strip() for line in re.split(r"(?<=[.!?])\s+", narration_text or "")
             if line.strip()]
    if not lines:
        return []
    total = audio_duration or 0.0
    if total <= 0:
        return []
    span = total / len(lines)
    groups = []
    for i, line in enumerate(lines):
        start = i * span
        end = min((i + 1) * span, total)
        words = tuple(WordCue(start=start, end=end, text=word) for word in line.split())
        groups.append(GroupCue(start=start, end=end, words=words))
    return groups


def build_caption_overlays(
    wav_path: Path,
    narration_text: str,
    audio_duration: float,
    out_dir: Path,
    *,
    accent_palette: tuple[str, ...] = DEFAULT_ACCENT_PALETTE,
    font_path: str | None = None,
    whisper_model: str = "base.en",
    gap_tight_ms: int = DEFAULT_GAP_TIGHT_MS,
    long_word_ms: int = DEFAULT_LONG_WORD_MS,
    max_group: int = DEFAULT_MAX_GROUP,
    cta_block: tuple[str, float, float] | None = None,
    segments: list[tuple[str, float, float]] | None = None,
) -> list[tuple[Path, float, float]]:
    """End-to-end: align -> filter -> pack -> render. Never returns empty when there is speech.

    `cta_block` is (text, start_s, end_s): the CTA is rendered as one static centred block with
    a subscribe button instead of word-synced groups, and any word group that falls inside that
    window is dropped so the two never overlap.

    `segments` is (text, start_s, end_s) per spoken beat. It is only used by the line-level
    fallback, which runs when word alignment is unavailable.

    A published short with no subtitles is worse than one with coarse subtitles, so when word
    alignment produces nothing the caller gets line-level captions instead of an empty list.
    """
    cues = align_words(wav_path, narration_text, model_size=whisper_model)
    cues = filter_usable_words(cues, audio_duration)
    if not cues:
        logger.warning("Word alignment produced no cues; using line-level captions instead")
        groups = build_line_level_cues(
            narration_text, segments=segments, audio_duration=audio_duration)
        if not groups:
            return []
        overlays = render_group_pngs(
            groups, out_dir, accent_palette=accent_palette, font_path=font_path)
        if cta_block:
            overlays = _apply_cta_block(overlays, cta_block, out_dir, font_path)
        return overlays
    groups = pack_groups(
        cues,
        gap_tight_ms=gap_tight_ms,
        long_word_ms=long_word_ms,
        max_group=max_group,
    )
    if not groups:
        return []
    # Never let a caption outlive the picture: the last group's tail can run past the audio
    # (and therefore past `-t`), which would leave an overlay enabled on a frame that does not
    # exist. Clamp to the track length and drop anything that starts after it.
    if audio_duration > 0:
        groups = [
            GroupCue(start=g.start, end=min(g.end, audio_duration), words=g.words)
            for g in groups
            if g.start < audio_duration
        ]

    overlays = render_group_pngs(
        groups, out_dir,
        accent_palette=accent_palette,
        font_path=font_path,
    )

    if cta_block:
        overlays = _apply_cta_block(overlays, cta_block, out_dir, font_path)

    return overlays


def _apply_cta_block(
    overlays: list[tuple],
    cta_block: tuple[str, float, float],
    out_dir: Path,
    font_path: str | None,
) -> list[tuple]:
    """Replace overlapping captions with one static centred CTA block."""
    cta_text, cta_start, cta_end = cta_block
    # Drop word groups that overlap the CTA window; the block replaces them.
    kept = [(p, s, e) for (p, s, e) in overlays if e <= cta_start or s >= cta_end]
    block_path = out_dir / "cta_block.png"
    if render_cta_block(cta_text, block_path, font_path=font_path):
        # "center" anchors the block to the middle of the frame, not the caption lane.
        kept.append((block_path, cta_start, cta_end, "center"))
    return kept


__all__ = [
    "DEFAULT_ACCENT_PALETTE",
    "DEFAULT_FONT_CANDIDATES",
    "DEFAULT_GAP_TIGHT_MS",
    "DEFAULT_LONG_WORD_MS",
    "DEFAULT_MAX_GROUP",
    "GroupCue",
    "SOFT_MUTE",
    "STOPWORDS",
    "WordCue",
    "align_words",
    "build_caption_overlays",
    "build_line_level_cues",
    "classify_word",
    "filter_usable_words",
    "find_gaps",
    "is_hallucination_token",
    "missing_caption_dependencies",
    "pack_groups",
    "render_cta_block",
    "render_group_pngs",
]
