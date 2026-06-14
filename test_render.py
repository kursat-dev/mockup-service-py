#!/usr/bin/env python3
"""
test_render.py — Local rendering test / preview tool.

Usage:
    # Render single-1 with a test artwork and save output
    python test_render.py --category single --index 1 \
        --template path/to/single-1.jpeg \
        --artwork  path/to/my_artwork.jpg \
        --output   output/test_single_1.jpg

    # Triple mockup (3 artworks required)
    python test_render.py --category triple --index 1 \
        --template path/to/triple-1.jpeg \
        --artwork  art1.jpg art2.jpg art3.jpg \
        --output   output/test_triple_1.jpg

    # Open result in default viewer automatically
    python test_render.py --category single --index 2 \
        --template path/to/single-2.jpeg \
        --artwork  art.jpg \
        --open

    # Show coordinate info without rendering
    python test_render.py --list

Options:
    --no-shadow         Disable inner shadow effect
    --no-lighting       Disable lighting multiply
    --glare             Enable glass glare (off by default)
    --quality INT       JPEG quality (default 93)
"""

import argparse
import json
import os
import sys
import subprocess
import platform

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
    """Returns (frames, calibrated, filename)"""
    cat_db = db.get(category, {})
    entry  = cat_db.get(index) or cat_db.get(str(int(index)))
    if entry:
        return entry.get("frames", []), entry.get("calibrated", False), entry.get("filename", "")
    return [], False, ""


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
            note     = data.get("_note", "")
            print(f"   [{idx}] {fn:30s} {n_frames} frame(s)  {cal}")
            if note:
                print(f"         note: {note}")
    print()


def open_file(path: str):
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", path])
    elif system == "Windows":
        os.startfile(path)
    else:
        subprocess.run(["xdg-open", path])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test mockup rendering locally")
    parser.add_argument("--category",   default="single",      help="Category: single, kitchen, triple, six")
    parser.add_argument("--index",      default="1",           help="Mockup index (number only)")
    parser.add_argument("--template",   help="Path to mockup template JPEG")
    parser.add_argument("--artwork",    nargs="+",             help="Path(s) to artwork image(s)")
    parser.add_argument("--output",     default="test_output.jpg", help="Output file path")
    parser.add_argument("--open",       action="store_true",   help="Open result after rendering")
    parser.add_argument("--no-shadow",  action="store_true",   help="Disable inner shadow")
    parser.add_argument("--no-lighting",action="store_true",   help="Disable lighting multiply")
    parser.add_argument("--glare",      action="store_true",   help="Enable glass glare")
    parser.add_argument("--quality",    type=int, default=93,  help="JPEG output quality (1-100)")
    parser.add_argument("--list",       action="store_true",   help="List all DB entries and exit")

    args = parser.parse_args()
    db   = load_db()

    if args.list:
        list_db(db)
        return

    if not args.template:
        parser.error("--template is required for rendering.")
    if not args.artwork:
        parser.error("--artwork is required for rendering.")

    # Validate paths
    if not os.path.exists(args.template):
        print(f"[ERROR] Template not found: {args.template}")
        sys.exit(1)
    for art in args.artwork:
        if not os.path.exists(art):
            print(f"[ERROR] Artwork not found: {art}")
            sys.exit(1)

    # Look up frames
    frames, calibrated, db_filename = lookup_frames(db, args.category, args.index)

    if not frames:
        print(f"[WARN] No coordinate entry found for category='{args.category}' index='{args.index}'.")
        print("       Using generic fallback regions. Use calibrate.html to set proper coordinates.")
    else:
        tag = "✅ calibrated" if calibrated else "⚠️  NOT calibrated (auto-detected, may be wrong)"
        print(f"[DB]   category='{args.category}' index='{args.index}' filename='{db_filename}' frames={len(frames)} — {tag}")

    # Read files
    with open(args.template, "rb") as f:
        template_bytes = f.read()

    product_bytes_list = []
    for art_path in args.artwork:
        with open(art_path, "rb") as f:
            product_bytes_list.append(f.read())

    if not frames:
        # Use inline default
        frames = [{
            "index": 1,
            "corners": {
                "top_left":     [260, 150],
                "top_right":    [740, 150],
                "bottom_right": [740, 830],
                "bottom_left":  [260, 830]
            }
        }]

    print(f"[Render] Starting — category={args.category}, index={args.index}, "
          f"frames={len(frames)}, artworks={len(product_bytes_list)}")
    print(f"         shadow={'on' if not args.no_shadow else 'off'}  "
          f"lighting={'on' if not args.no_lighting else 'off'}  "
          f"glare={'on' if args.glare else 'off'}  "
          f"quality={args.quality}")

    import time
    t0 = time.time()

    result = warp_perspective_cv(
        template_bytes=template_bytes,
        product_bytes_list=product_bytes_list,
        frames=frames,
        inner_shadow=not args.no_shadow,
        glass_glare=args.glare,
        paper_texture=False,
        lighting_multiply=not args.no_lighting
    )

    ms = int((time.time() - t0) * 1000)
    print(f"[Render] Done in {ms}ms — output size {len(result):,} bytes")

    # Patch quality into output (renderer uses hardcoded 93 — this is for the test script preview)
    # Re-save with custom quality if different
    if args.quality != 93:
        from PIL import Image
        import io
        pil = Image.open(io.BytesIO(result))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=args.quality, optimize=True)
        result = buf.getvalue()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        f.write(result)

    print(f"[OK]    Saved → {os.path.abspath(args.output)}")

    if args.open:
        open_file(args.output)


if __name__ == "__main__":
    main()
