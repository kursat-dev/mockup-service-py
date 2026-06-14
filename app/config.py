import os

PORT = int(os.getenv("PORT", 5111))
HOST = os.getenv("HOST", "0.0.0.0")
ALLOW_PERSPECTIVE_FALLBACK = os.getenv("ALLOW_PERSPECTIVE_FALLBACK", "false").lower() == "true"

# ── API Key Protection ──────────────────────────────────────────────────────
# If set, /render requires X-API-Key header matching this value.
# If empty, requests are allowed without authentication (local development).
MOCKUP_API_KEY = os.getenv("MOCKUP_API_KEY", "")

# ── Template directory ──────────────────────────────────────────────────────
# Base directory for internal mockup templates.
# In Docker: /app/templates   |   Locally: relative to project root
TEMPLATES_DIR = os.getenv("TEMPLATES_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))
