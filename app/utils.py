import hashlib

def calculate_sha256(buffer: bytes) -> str:
    """Calculate SHA-256 hash of image buffer to dynamically match coordinates."""
    return hashlib.sha256(buffer).hexdigest()
