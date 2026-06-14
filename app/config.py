import os

PORT = int(os.getenv("PORT", 5111))
HOST = os.getenv("HOST", "0.0.0.0")
ALLOW_PERSPECTIVE_FALLBACK = os.getenv("ALLOW_PERSPECTIVE_FALLBACK", "false").lower() == "true"
