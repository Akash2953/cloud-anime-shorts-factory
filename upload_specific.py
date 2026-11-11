# Simple helper to upload the specific output_part1.mp4 file
# Uses existing publish.py functions

import os
from config import load_dotenv
from publish import exchange_refresh_token, upload_video

def upload_part1():
    load_dotenv()

    # Get access token
    access_token = exchange_refresh_token(
        os.getenv("YOUTUBE_CLIENT_ID"),
        os.getenv("YOUTUBE_CLIENT_SECRET"),
        os.getenv("YOUTUBE_REFRESH_TOKEN")
    )

    if not access_token:
        print("Failed to get YouTube access token")
        return False

    # Upload Part 1 video
    title = "Anime Short #1/3 — reuse-lantern-kaito"
    description = """# Part 1 of 3

# Narration:
1. Neon rain swallows a city of endless towers.
2. She stands at the edge, hunting her ghosts.
3. Her amber eyes burn with a sister's promise.
4. He traced escape routes that she now follows.
5. A paper lantern bears his name in fading ink.
6. Roof to roof she flies through lantern light.
7. Dark oily alleys reflect pink neon hunting grounds.
8. His scratched data chip surfaces in the muck.
9. Three hunter drones lock red eyes on her.
10. Code-lanterns orbit, shredding steel drones with light.
11. Biometrics confirm him. Sub-level four.
12. The platform drops, lanterns spiralling down behind her."""

    tags = ["anime", "shorts", "storytime", "mystery", "lantern", "rainy city", "vertical video"]
    privacy_status = os.getenv("PUBLISH_PRIVACY", "private")

    # Read and upload the video
    with open("build/output_part1.mp4", "rb") as f:
        video_bytes = f.read()

    video_id = upload_video(video_bytes, title, description, access_token,
                           privacy_status=privacy_status, tags=tags)

    if video_id:
        print(f"Upload successful! Video ID: {video_id}")
        print(f"URL: https://youtube.com/watch?v={video_id}")
        return True
    else:
        print("Upload failed")
        return False

if __name__ == "__main__":
    upload_part1()