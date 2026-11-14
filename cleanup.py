"""Delete the run's own artefacts once the short is published.

Everything the pipeline produces is disposable after a successful upload: the frames, the caption
PNGs, the narration and music beds, the thumbnail and the encoded MP4. Nothing deleted here is
needed to publish again — YouTube holds the video and `state.json` holds the story — so before
this existed a runner (or a laptop) kept ~30 MB per run, and `output_short.mp4` sat in the repo
root looking like a deliverable while actually being yesterday's upload.

Only the paths listed in `ARTIFACTS` are removed. `build/` can hold unrelated work, so nothing
here globs or walks it: an unrecognised directory is left exactly where it is.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# One entry per artefact the pipeline itself writes, in a normal run.
ARTIFACTS: tuple[str, ...] = (
    "output_short.mp4",
    "build/frames",
    "build/captions",
    "build/narration.wav",
    "build/music_bed.wav",
    "build/thumbnail.jpg",
    "build/concat.txt",
)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def remove_artifacts(root: Path = Path("."), *, artifacts: tuple[str, ...] = ARTIFACTS,
                     keep: tuple[str, ...] = ()) -> tuple[list[Path], int]:
    """Remove the run's artefacts under `root`. Returns (removed paths, bytes freed).

    Unlisted paths are never touched, and a missing artefact is not an error: a run that failed
    before rendering simply has less to delete.
    """
    keep_set = {Path(item) for item in keep}
    removed: list[Path] = []
    freed = 0
    for name in artifacts:
        path = root / name
        if not path.exists() or Path(name) in keep_set:
            continue
        try:
            freed += _size(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:  # a locked file must not fail an upload that already succeeded
            logger.warning("Cleanup: could not remove %s (%s)", path, exc)
            continue
        removed.append(path)
    if removed:
        logger.info("Cleanup: removed %d artefact(s), %.1f MB freed",
                    len(removed), freed / 1e6)
    return removed, freed
