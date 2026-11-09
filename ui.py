"""Web UI for manual short upload triggers.

Usage:
    python -m ui  # Run the web server
    or
    from ui import app; app.run()  # Import as module
"""

import logging
import sys
from datetime import date
from pathlib import Path

from flask import Flask, redirect, render_template_string, url_for

from config import Limits
from orchestrator import run_pipeline
from state import parse_state

app = Flask(__name__)
app.config["DEBUG"] = False

logger = logging.getLogger(__name__)

# Pipeline tracking
pipeline_steps = {
    "story": "Generating story",
    "images": "Creating images",
    "render": "Rendering video",
    "upload": "Uploading to storage",
    "youtube": "Uploading to YouTube",
    "complete": "Process completed"
}
current_step = "idle"
pipeline_status = "idle"

log_messages = []


def add_log(message: str, level: str = "info"):
    """Add a log message to the cache."""
    log_messages.append({"type": level, "message": message})
    if len(log_messages) > 20:
        log_messages.pop(0)


def get_limits():
    """Get the Limits configuration."""
    return Limits()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cloud Anime Shorts Factory</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #ff6b6b; font-size: 2rem; margin-bottom: 24px; text-align: center; }
        .card { background: rgba(255,255,255,.1); border-radius: 12px; padding: 20px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,.2); }
        .label { color: #888; font-size: .8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; display: block; }
        .value { font-size: 1.4rem; font-weight: bold; }
        .badge { font-size: .7rem; margin-left: 8px; padding: 2px 8px; border-radius: 999px; vertical-align: middle; }
        .badge.ok { background: #238636; color: #fff; }
        .badge.full { background: #6e2c2c; color: #fff; }
        button {
            width: 100%; padding: 14px; font-size: 1rem; font-weight: bold; color: #fff;
            background: linear-gradient(135deg, #ff6b6b, #ee5a5a); border: 0; border-radius: 8px; cursor: pointer;
        }
        button:disabled { background: #555; cursor: not-allowed; }
        .link { display: block; text-align: center; margin-top: 12px; color: #888; background: none; font-weight: normal; text-decoration: underline; }
        ul { list-style: none; margin: 0; padding: 0; }
        li { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.1); }
        li:last-child { border-bottom: 0; }
        .log { background: #0d1117; border-radius: 8px; padding: 12px; font-family: ui-monospace, monospace; font-size: .8rem; max-height: 200px; overflow-y: auto; border-left: 3px solid #ff6b6b; }
        .log div { margin-bottom: 4px; word-break: break-all; }
        .log .error { color: #f85149; }
        .log .success { color: #51cf66; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Anime Shorts Factory</h1>

        <div class="card">
            <div class="label">Daily upload quota</div>
            <div class="value">
                <span id="uploads">{{ uploads + ' / ' + max_uploads }}</span>
                <span class="badge" id="quota-badge">{{ quota_full ? 'UPLOADED' : 'AVAILABLE' }}</span>
            </div>
        </div>
        <div class="card">
            <div class="label">Image attempts today</div>
            <div class="value" id="images">{{ images + ' / ' + max_images }}</div>
        </div>
        <button id="upload-btn" onclick="uploadShort()">Generate & Upload Short</button>
        <a class="link" onclick="showQuota()">Quota info</a>
        <div class="card" id="quota-card" style="display:none;">
            <div class="label">Hard limits</div>
            <ul>
                <li><span>Max images per short</span><b>6</b></li>
                <li><span>Max uploads per day</span><b>1</b></li>
                <li><span>Max image attempts per day</span><b>8</b></li>
                <li><span>OpenRouter free daily limit</span><b>50</b></li>
            </ul>
            <div class="card">
                <div class="label">Current usage</div>
                <ul>
                    <li><span>Images attempted today</span><b id="current-images">0/8</b></li>
                    <li><span>Stories requested today</span><b id="current-stories">0/50</b></li>
                    <li><span>Uploads today</span><b id="current-uploads">0/1</b></li>
                </ul>
            </div>
            <button class="link" onclick="hideQuota()">Back to dashboard</button>
        </div>
    </div>

    <script>
        function uploadShort() {
            fetch('/api/upload', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.ok) {
                        alert('Upload completed successfully!');
                    } else {
                        alert('Error: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(error => {
                    alert('Request failed: ' + error);
                });
        }

        function showQuota() {
            document.getElementById('quota-card').style.display = 'block';
            document.querySelector('.container').style.display = 'none';
        }

        function hideQuota() {
            document.getElementById('quota-card').style.display = 'none';
            document.querySelector('.container').style.display = 'block';
        }

        // Poll status every 3 seconds
        setInterval(() => {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('uploads').textContent =
                        data.uploads + ' / ' + data.max_uploads;
                    document.getElementById('quota-badge').className =
                        data.quota_full ? 'badge full' : 'badge ok';
                    document.getElementById('quota-badge').textContent =
                        data.quota_full ? 'UPLOADED' : 'AVAILABLE';
                    document.getElementById('images').textContent =
                        data.images + ' / ' + data.max_images;
                    document.getElementById('current-images').textContent =
                        data.images + '/' + data.max_images;
                    document.getElementById('current-stories').textContent =
                        data.stories + '/' + data.limits.openrouter_free_daily_limit;
                    document.getElementById('current-uploads').textContent =
                        data.uploads + '/' + data.max_uploads;
                });
        }, 3000);
    </script>
</body>
</html>"""

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and not path.startswith('api/'):
        return app.send_static_file('index.html')
    return '', 404

@app.route('/api/status')
def api_status():
    from config import Limits
    from state import parse_state
    from pathlib import Path
    state_path = Path("state.json")
    state_raw = state_path.read_text(encoding="utf-8") if state_path.exists() else None
    state = parse_state(state_raw)
    today = date.today().isoformat()
    daily = state.get("daily", {}).get(today, {"uploads": 0, "image_attempts": 0})
    limits = Limits()
    uploads = daily.get("uploads", 0)
    return {
        "uploads": uploads,
        "max_uploads": limits.max_daily_uploads,
        "images": daily.get("image_attempts", 0),
        "max_images": limits.max_image_attempts_per_day,
        "quota_full": uploads >= limits.max_daily_uploads,
        "running": False,
        "logs": log_messages,
        "limits": {
            "max_images_per_short": limits.max_images_per_short,
            "max_daily_uploads": limits.max_daily_uploads,
            "max_image_attempts_per_day": limits.max_image_attempts_per_day,
            "openrouter_free_daily_limit": limits.openrouter_free_daily_limit,
        },
        "stories": state.get("stories_requested_today", 0),
    }

@app.route('/api/upload', methods=['POST'])
def api_upload():
    global running
    if running:
        return {"ok": False, "error": "already running"}, 409
    running = True
    add_log("Manual upload triggered via UI", "info")
    try:
        run_pipeline()
        add_log("Upload completed successfully", "success")
        return {"ok": True}
    except Exception as exc:
        logger.error("Upload failed: %s", exc)
        add_log(f"Upload failed: {exc}", "error")
        return {"ok": False, "error": str(exc)}, 500
    finally:
        running = False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)