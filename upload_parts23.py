import json
import os
from pathlib import Path

from config import load_dotenv
from publish import exchange_refresh_token, upload_video

load_dotenv()
token = exchange_refresh_token(
    os.getenv("YOUTUBE_CLIENT_ID"),
    os.getenv("YOUTUBE_CLIENT_SECRET"),
    os.getenv("YOUTUBE_REFRESH_TOKEN"),
)
if not token:
    raise SystemExit("Token exchange failed")

state = json.loads(Path("state.json").read_text(encoding="utf-8"))
series = state["series"]
total = len(series["parts"])
sid = series.get("id", "new")
privacy = os.getenv("PUBLISH_PRIVACY", "private")
tags = ["anime", "shorts", "storytime", "mystery", "lantern", "rainy city", "vertical video"]

for part_num in (2, 3):
    shots = series["parts"][part_num - 1]
    title = f"Anime Short #{part_num}/{total} - {sid}"
    desc = [f"# Part {part_num} of {total}", "", "# Narration:"]
    for i, shot in enumerate(shots, 1):
        desc.append(f"{i}. {shot.get('narration', '')}")
    path = Path(f"build/output_part{part_num}.mp4")
    print(f"Uploading {path} ({path.stat().st_size/1024/1024:.1f} MB), privacy={privacy}")
    vid = upload_video(path.read_bytes(), title, "\n".join(desc), token,
                       privacy_status=privacy, tags=tags)
    print("OK: https://youtube.com/watch?v=" + vid if vid else "FAILED")