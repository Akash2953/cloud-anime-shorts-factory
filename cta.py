"""End-of-short call to action.

A Short that ends on the last story beat asks the viewer for nothing, so the next part has to
find them by luck. One spoken line over a held final frame converts that ending into a reason
to subscribe — and it is the only place in the video where asking is not intrusive.

The CTA is deliberately *not* narrated like the story. It is spoken by a brighter voice, a
little faster, with exclamation marks, because a line delivered in the same measured
storyteller tone reads as one more story beat and the viewer stops listening. It is also
displayed as one static block rather than word-synced captions, with the word "subscribe"
replaced by a subscribe button, so the ask is legible in a single glance.
"""

from __future__ import annotations

# Spoken over the held last frame. Short, second person, exclamation marks: the punctuation is
# what makes Kokoro lift the pitch, so it is doing real work here, not decoration.
CTA_MIDDLE = "Part {next} is already waiting for you! Subscribe so you don't miss it!"
CTA_LAST = "There are more stories waiting for you! Subscribe to keep them coming!"
CTA_SINGLE = "There are more stories waiting for you! Subscribe to keep them coming!"

# A brighter, quicker voice than the narration. Measured on the CTA line: af_bella has a
# spectral centroid of 1909 Hz against af_heart's 1772, so it reads as lifted and upbeat, while
# af_sky (1041 Hz) is actually darker and would sound *less* happy than the narrator. The speed
# bump keeps it from sounding like a slowed-down narrator.
CTA_VOICE = "af_bella"
CTA_SPEED = 1.15

# The word the block renderer swaps for a subscribe button.
SUBSCRIBE_WORD = "subscribe"


def cta_line(part: int, total_parts: int) -> str:
    """The spoken CTA for this short.

    A middle part points at the next part by number, because that is the specific thing the
    viewer would otherwise miss. The final part of an arc (and a single-part story) asks for a
    subscription to the channel instead, since there is no next part to promise.
    """
    if total_parts > 1 and part < total_parts:
        return CTA_MIDDLE.format(next=part + 1)
    if total_parts > 1:
        return CTA_LAST
    return CTA_SINGLE


def cta_shot(part: int, total_parts: int, seconds: float = 4.0):
    """Build the CTA as a Shot so it flows through narration, captions and timing unchanged.

    Imported lazily to keep this module free of a scenes import cycle.
    """
    from scenes import Shot

    return Shot(
        narration=cta_line(part, total_parts),
        shot_size="wide",
        place="the last frame, held",
        action="the final image holds while the narrator speaks to the viewer",
        story_object="no object",
        narrative_role="call_to_action",
        camera="static",
        character=False,
        seconds=seconds,
    )


__all__ = [
    "CTA_LAST",
    "CTA_MIDDLE",
    "CTA_SINGLE",
    "CTA_SPEED",
    "CTA_VOICE",
    "SUBSCRIBE_WORD",
    "cta_line",
    "cta_shot",
]
