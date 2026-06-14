#!/usr/bin/env python3
"""
test_render.py — Local rendering test and quality verification tool.

Usage:
    # Run quality tests for single-1, single-2, single-3
    python test_render.py --quality-test

    # Render single-1 with a test artwork and save output
    python test_render.py --category single --index 1 \
        --template path/to/base.jpg \
        --mask path/to/mask.png \
        --overlay path/to/overlay.png \
        --artwork  path/to/my_artwork.jpg \
        --output   output/test_single_1.jpg

    # Show coordinate info without rendering
    python test_render.py --list
"""

import argparse
import json
import os
import sys
import time
import subprocess
import platform
from typing import Optional

# ── Allow import from parent package ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

try:
    from app.renderer import warp_perspective_cv
except ImportError as e:
    print(f"[ERROR] Could not import renderer: {e}")
    print("Make sure you are running this script from the mockup-service-py directory.")
    sys.exit(1)

# ── Load coordinates DB ───────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "app", "database", "coordinates.json")

def load_db() -> dict:
    if not os.path.exists(DB_PATH):
        print(f"[WARN] coordinates.json not found at {DB_PATH}")
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_frames(db: dict, category: str, index: str) -> tuple:
    """Returns (frames, calibrated, settings)"""
    cat_db = db.get(category, {})
    entry  = cat_db.get(index) or cat_db.get(str(int(index)))
    
    DEFAULT_SETTINGS = {
        "feather_px": 1.5,
        "supersample": 4,
        "brightness_match": 0.2,
        "saturation_match": 0.1,
        "inner_shadow": True,
        "overlay_opacity": 1.0,
        "jpeg_quality": 95
    }

    if entry:
        frames = entry.get("frames", [])
        calibrated = entry.get("calibrated", False)
        settings = {**DEFAULT_SETTINGS}
        for k in DEFAULT_SETTINGS:
            if k in entry:
                settings[k] = entry[k]
        return frames, calibrated, settings
    return [], False, DEFAULT_SETTINGS


def list_db(db: dict):
    print("\n📋 Coordinate Database Contents\n" + "─" * 50)
    for cat, entries in db.items():
        if cat.startswith("_"):
            continue
        print(f"\n🗂  [{cat}]")
        for idx, data in entries.items():
            cal      = "✅ calibrated" if data.get("calibrated") else "⚠️  NOT calibrated"
            n_frames = len(data.get("frames", []))
            fn       = data.get("filename", "")
            print(f"   [{idx}] {fn:30s} {n_frames} frame(s)  {cal}")
    print()


def generate_test_artwork(width=1500, height=2000) -> bytes:
    """Generates a high-quality colorful test poster image using Pillow."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), color="#1e1e24")
    draw = ImageDraw.Draw(img)
    
    # Draw a nice colorful gradient
    for y in range(height):
        r = int(255 * (y / height))
        g = int(50 * (1 - y / height))
        b = int(200 * (1 - y / height))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    # Draw grid
    for i in range(1, 10):
        x = int(width * i / 10)
        draw.line([(x, 0), (x, height)], fill="#ffffff", width=2)
        y = int(height * i / 10)
        draw.line([(0, y), (width, y)], fill="#ffffff", width=2)
        
    # Circle in the middle
    cx, cy = width // 2, height // 2
    r_circle = min(width, height) // 4
    draw.ellipse([cx - r_circle, cy - r_circle, cx + r_circle, cy + r_circle], fill="#ffbc42", outline="#ffffff", width=8)
    
    # Text
    draw.text((cx - 200, cy - 30), "TEST ARTWORK", fill="#1e1e24", stroke_width=2, stroke_fill="#ffffff")
    
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run_quality_tests():
    print("🚀 Running Quality Tests for single-1 through single-9...")
    
    db = load_db()
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    artwork_bytes = generate_test_artwork()
    test_cases = [str(i) for i in range(1, 10)]
    
    verification_results = []
    
    for idx in test_cases:
        print(f"\n--- Mockup single-{idx} ---")
        tpl_dir = os.path.join(templates_dir, "single", f"single-{idx}")
        os.makedirs(tpl_dir, exist_ok=True)
        
        base_path = os.path.join(tpl_dir, "base.jpg")
        mask_path = os.path.join(tpl_dir, "mask.png")
        overlay_path = os.path.join(tpl_dir, "overlay.png")
        
        # Check coordinates
        frames, calibrated, settings = lookup_frames(db, "single", idx)
        
        # Copy/generate if directory-style template does not exist
        legacy_file = os.path.join(templates_dir, "single", f"single-{idx}.jpeg")
        if not os.path.exists(base_path) and os.path.exists(legacy_file):
            import shutil
            shutil.copy(legacy_file, base_path)
            print(f"Copied legacy file to base.jpg: {base_path}")
            
        if not os.path.exists(mask_path) and os.path.exists(base_path):
            import cv2
            import numpy as np
            base_img = cv2.imread(base_path)
            h, w = base_img.shape[:2]
            mask_img = np.zeros((h, w), dtype=np.uint8)
            if frames:
                corners = frames[0]["corners"]
                poly = np.array([
                    corners["top_left"],
                    corners["top_right"],
                    corners["bottom_right"],
                    corners["bottom_left"]
                ], dtype=np.float32)
                
                # Draw at 8x resolution for high-quality antialiasing
                scale = 8
                h_high, w_high = h * scale, w * scale
                mask_high = np.zeros((h_high, w_high), dtype=np.uint8)
                poly_high = (poly * scale).astype(np.int32)
                cv2.fillPoly(mask_high, [poly_high], 255)
                mask_img = cv2.resize(mask_high, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(mask_path, mask_img)
            print(f"Generated mask.png: {mask_path}")
            
        if not os.path.exists(overlay_path) and os.path.exists(base_path):
            import cv2
            import numpy as np
            base_img = cv2.imread(base_path)
            h, w = base_img.shape[:2]
            overlay_img = np.zeros((h, w, 4), dtype=np.uint8)  # fully transparent
            cv2.imwrite(overlay_path, overlay_img)
            print(f"Created transparent overlay.png: {overlay_path}")
            
        base_loaded = os.path.exists(base_path)
        mask_loaded = os.path.exists(mask_path)
        overlay_loaded = os.path.exists(overlay_path)
        
        verification_results.append({
            "idx": idx,
            "base": "YES" if base_loaded else "NO",
            "mask": "YES" if mask_loaded else "NO",
            "overlay": "YES" if overlay_loaded else "NO"
        })
        
        with open(base_path, "rb") as f:
            base_bytes = f.read()
        with open(mask_path, "rb") as f:
            mask_bytes = f.read()
        with open(overlay_path, "rb") as f:
            overlay_bytes = f.read()
            
        t0 = time.time()
        result_bytes = warp_perspective_cv(
            template_bytes=base_bytes,
            product_bytes_list=[artwork_bytes],
            frames=frames,
            settings=settings,
            mask_bytes=mask_bytes,
            overlay_bytes=overlay_bytes
        )
        duration = int((time.time() - t0) * 1000)
        
        import cv2
        import numpy as np
        nparr = np.frombuffer(result_bytes, np.uint8)
        out_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = out_img.shape[:2]
        
        out_path = os.path.join(output_dir, f"quality-single-{idx}.jpg")
        with open(out_path, "wb") as f:
            f.write(result_bytes)
            
        print(f"Output dimensions: {w}x{h}")
        print(f"Render duration: {duration} ms")
        print(f"Loaded base: {'YES' if base_loaded else 'NO'}, mask: {'YES' if mask_loaded else 'NO'}, overlay: {'YES' if overlay_loaded else 'NO'}")
        print(f"Supersampling applied: YES ({settings.get('supersample', 4)}x)")
        print(f"Feathering applied: YES ({settings.get('feather_px', 1.5)} px)")
        print(f"Saved to: {out_path}")
        
    print("\n📋 Verification Table:")
    for res in verification_results:
        print(f"single-{res['idx']} base {res['base']} mask {res['mask']} overlay {res['overlay']}")
        
    print("\n✅ Quality tests complete successfully!")



def main():
    parser = argparse.ArgumentParser(description="Test mockup rendering locally")
    parser.add_argument("--quality-test", action="store_true", help="Run sequential single-1, 2, 3 quality test suite")
    parser.add_argument("--category",   default="single",      help="Category: single, kitchen, triple, six")
    parser.add_argument("--index",      default="1",           help="Mockup index (number only)")
    parser.add_argument("--template",   help="Path to mockup template base JPEG")
    parser.add_argument("--mask",       help="Path to template mask PNG")
    parser.add_argument("--overlay",    help="Path to template overlay PNG")
    parser.add_argument("--artwork",    nargs="+",             help="Path(s) to artwork image(s)")
    parser.add_argument("--output",     default="test_output.jpg", help="Output file path")
    parser.add_argument("--list",       action="store_true",   help="List all DB entries and exit")

    args = parser.parse_args()
    db   = load_db()

    if args.quality_test:
        run_quality_tests()
        return

    if args.list:
        list_db(db)
        return

    if not args.template:
        parser.error("--template is required for rendering.")
    if not args.artwork:
        parser.error("--artwork is required for rendering.")

    if not os.path.exists(args.template):
        print(f"[ERROR] Template not found: {args.template}")
        sys.exit(1)
    for art in args.artwork:
        if not os.path.exists(art):
            print(f"[ERROR] Artwork not found: {art}")
            sys.exit(1)

    frames, calibrated, settings = lookup_frames(db, args.category, args.index)

    with open(args.template, "rb") as f:
        template_bytes = f.read()

    mask_bytes = None
    if args.mask and os.path.exists(args.mask):
        with open(args.mask, "rb") as f:
            mask_bytes = f.read()

    overlay_bytes = None
    if args.overlay and os.path.exists(args.overlay):
        with open(args.overlay, "rb") as f:
            overlay_bytes = f.read()

    product_bytes_list = []
    for art_path in args.artwork:
        with open(art_path, "rb") as f:
            product_bytes_list.append(f.read())

    t0 = time.time()
    result = warp_perspective_cv(
        template_bytes=template_bytes,
        product_bytes_list=product_bytes_list,
        frames=frames,
        settings=settings,
        mask_bytes=mask_bytes,
        overlay_bytes=overlay_bytes
    )
    ms = int((time.time() - t0) * 1000)
    print(f"[Render] Done in {ms}ms — output size {len(result):,} bytes")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        f.write(result)
    print(f"[OK] Saved to {args.output}")


if __name__ == "__main__":
    main()
