from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.responses import Response, JSONResponse
import json
import os
import time
from typing import Optional

from app.renderer import warp_perspective_cv
from app.config import PORT, HOST, MOCKUP_API_KEY, TEMPLATES_DIR

app = FastAPI(
    title="Etsy Mockup Render Service",
    description="Zero-cost, deterministic computer-vision mockup rendering engine.",
    version="8.0.0"
)

# ── Load coordinate database once at startup ────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "database", "coordinates.json")
coordinates_db: dict = {}
coordinates_loaded: bool = False

if os.path.exists(DB_PATH):
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            coordinates_db = json.load(f)
        total = sum(len(v) for k, v in coordinates_db.items() if not k.startswith("_"))
        categories = sum(1 for k in coordinates_db if not k.startswith("_"))
        print(f"[Startup] Coordinate DB loaded — {total} mockup entries across {categories} categories.")
        coordinates_loaded = True
    except Exception as e:
        print(f"[Startup ERROR] Failed to load coordinate DB: {e}")
else:
    print(f"[Startup WARNING] Coordinate DB not found at {DB_PATH}. Only fallback defaults will be used.")

# ── Validate internal templates at startup ───────────────────────────────────
internal_templates: dict = {}  # { "single": { "1": { "type": "directory", "path": "..." } } }

if os.path.exists(TEMPLATES_DIR):
    categories = ["single", "kitchen", "triple", "six"]
    for cat in categories:
        internal_templates[cat] = {}
        cat_dir = os.path.join(TEMPLATES_DIR, cat)
        if not os.path.exists(cat_dir):
            continue
        for item in os.listdir(cat_dir):
            item_path = os.path.join(cat_dir, item)
            if os.path.isdir(item_path):
                if item.startswith(f"{cat}-"):
                    index_str = item.split("-")[-1]
                    if (os.path.exists(os.path.join(item_path, "base.jpg")) and
                        os.path.exists(os.path.join(item_path, "mask.png")) and
                        os.path.exists(os.path.join(item_path, "overlay.png"))):
                        internal_templates[cat][index_str] = {
                            "type": "directory",
                            "path": item_path
                        }
                        print(f"[Startup] ✅ Directory Template found: {cat}/{item}")
            elif os.path.isfile(item_path):
                if item.startswith(f"{cat}-") and (item.endswith(".jpeg") or item.endswith(".jpg")):
                    name_without_ext = os.path.splitext(item)[0]
                    index_str = name_without_ext.split("-")[-1]
                    internal_templates[cat][index_str] = {
                        "type": "file",
                        "path": item_path
                    }
                    print(f"[Startup] ✅ File Template found: {cat}/{item}")

for cat, templates in internal_templates.items():
    found = len(templates)
    print(f"[Startup] Template category '{cat}': {found} templates found on disk")


# ── Default frame regions (fallback when no calibrated entry exists) ─────────
def get_default_regions(category: str) -> list:
    """Ratio-based fallback regions. Values are PIXELS on a 1000×1000 canvas."""
    if category in ("single", "kitchen"):
        return [{"index": 1, "corners": {
            "top_left": [260, 150], "top_right": [740, 150],
            "bottom_right": [740, 830], "bottom_left": [260, 830]
        }}]
    elif category == "triple":
        return [
            {"index": 1, "corners": {"top_left": [100, 180], "top_right": [350, 180], "bottom_right": [350, 780], "bottom_left": [100, 780]}},
            {"index": 2, "corners": {"top_left": [375, 180], "top_right": [625, 180], "bottom_right": [625, 780], "bottom_left": [375, 780]}},
            {"index": 3, "corners": {"top_left": [650, 180], "top_right": [900, 180], "bottom_right": [900, 780], "bottom_left": [650, 780]}},
        ]
    elif category == "six":
        return [
            {"index": 1, "corners": {"top_left": [100, 100], "top_right": [340, 100], "bottom_right": [340, 470], "bottom_left": [100, 470]}},
            {"index": 2, "corners": {"top_left": [380, 100], "top_right": [620, 100], "bottom_right": [620, 470], "bottom_left": [380, 470]}},
            {"index": 3, "corners": {"top_left": [660, 100], "top_right": [900, 100], "bottom_right": [900, 470], "bottom_left": [660, 470]}},
            {"index": 4, "corners": {"top_left": [100, 530], "top_right": [340, 530], "bottom_right": [340, 900], "bottom_left": [100, 900]}},
            {"index": 5, "corners": {"top_left": [380, 530], "top_right": [620, 530], "bottom_right": [620, 900], "bottom_left": [380, 900]}},
            {"index": 6, "corners": {"top_left": [660, 530], "top_right": [900, 530], "bottom_right": [900, 900], "bottom_left": [660, 900]}},
        ]
    return []


def lookup_frames(category: str, mockup_index: str) -> tuple[list, bool, dict]:
    """
    Returns (frames, is_calibrated, settings).
    Lookup order:
      1. coordinates_db[category][mockup_index]   ← canonical key
      2. coordinates_db[category][str(int(mockup_index))] ← int normalise
      3. Default ratio-based fallback
    """
    cat_db = coordinates_db.get(category, {})
    key = str(mockup_index)
    normalized_key = str(int(mockup_index)) if mockup_index.isdigit() else key

    entry = cat_db.get(key) or cat_db.get(normalized_key)

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
        # Load settings
        settings = {**DEFAULT_SETTINGS}
        for k in DEFAULT_SETTINGS:
            if k in entry:
                settings[k] = entry[k]
        return frames, calibrated, settings

    # Fallback
    return get_default_regions(category), False, DEFAULT_SETTINGS


def get_template_assets(category: str, mockup_index: str) -> Optional[dict]:
    """
    Returns a dict containing file paths for 'base', 'mask', and 'overlay'
    if three-asset template directory is found, or 'file' if legacy file is found.
    """
    cat_templates = internal_templates.get(category, {})
    key = str(mockup_index)
    normalized_key = str(int(mockup_index)) if mockup_index.isdigit() else key
    
    tpl = cat_templates.get(key) or cat_templates.get(normalized_key)
    if not tpl:
        return None
        
    if tpl["type"] == "directory":
        return {
            "type": "directory",
            "base": os.path.join(tpl["path"], "base.jpg"),
            "mask": os.path.join(tpl["path"], "mask.png"),
            "overlay": os.path.join(tpl["path"], "overlay.png")
        }
    else:
        return {
            "type": "file",
            "file": tpl["path"]
        }


# ── API Key middleware ────────────────────────────────────────────────────────
def verify_api_key(request: Request):
    """Verify API key if MOCKUP_API_KEY is configured."""
    if not MOCKUP_API_KEY:
        return  # No auth required in dev mode

    provided_key = request.headers.get("X-API-Key", "")
    if provided_key != MOCKUP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    total = sum(len(v) for k, v in coordinates_db.items() if not k.startswith("_"))
    template_count = sum(len(v) for v in internal_templates.values())
    return {
        "success": True,
        "message": "Mockup Render Service v8.0 — CV Engine",
        "templatesLoaded": total,
        "internalTemplates": template_count,
        "engine": "OpenCV + Pillow (100% deterministic, 0% AI cost)"
    }


@app.get("/health")
async def health():
    template_counts = {}
    for category, templates in internal_templates.items():
        template_counts[category] = len(templates)

    return {
        "status": "ok",
        "service": "mockup-service-py",
        "renderer": "opencv",
        "templates": template_counts,
        "coordinatesLoaded": coordinates_loaded
    }


@app.get("/db")
async def list_db():
    """List all coordinate entries in the database (for debugging)."""
    result = {}
    for cat, entries in coordinates_db.items():
        if cat.startswith("_"):
            continue
        result[cat] = {}
        for idx, data in entries.items():
            result[cat][idx] = {
                "filename": data.get("filename", ""),
                "calibrated": data.get("calibrated", False),
                "frameCount": len(data.get("frames", [])),
                "note": data.get("_note", "")
            }
    return result


@app.post("/render")
async def render(
    request: Request,
    image_1: UploadFile = File(...),
    mockup_template: Optional[UploadFile] = File(None),
    image_2: Optional[UploadFile] = File(None),
    image_3: Optional[UploadFile] = File(None),
    image_4: Optional[UploadFile] = File(None),
    image_5: Optional[UploadFile] = File(None),
    image_6: Optional[UploadFile] = File(None),
    category: str = Form("single"),
    mockup_index: str = Form("1"),
    total_mockups: str = Form("1"),
    title: Optional[str] = Form(""),
    image_notes: Optional[str] = Form("")
):
    # ── API Key check ────────────────────────────────────────────────────────
    verify_api_key(request)

    start_time = time.time()

    # ── Structured logging ───────────────────────────────────────────────────
    print(f"[Render] category={category} mockup_index={mockup_index} total={total_mockups}")

    allowed = ["single", "triple", "six", "kitchen"]
    if category not in allowed:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "errorType": "RENDER_ERROR",
                "message": f"Invalid category '{category}'. Must be one of: {', '.join(allowed)}"
            }
        )

    try:
        # ── Resolve template: internal first, uploaded fallback ──────────────
        base_bytes = None
        mask_bytes = None
        overlay_bytes = None
        template_source = "unknown"

        assets = get_template_assets(category, mockup_index)
        if assets:
            if assets["type"] == "directory":
                with open(assets["base"], "rb") as f:
                    base_bytes = f.read()
                with open(assets["mask"], "rb") as f:
                    mask_bytes = f.read()
                with open(assets["overlay"], "rb") as f:
                    overlay_bytes = f.read()
                template_source = os.path.dirname(assets["base"])
                print(f"[Render] Using three-asset template from {category}/{category}-{mockup_index}")
            else:
                with open(assets["file"], "rb") as f:
                    base_bytes = f.read()
                template_source = assets["file"]
                print(f"[Render] Using fallback single-file template {category}/{os.path.basename(assets['file'])}")
        elif mockup_template is not None:
            base_bytes = await mockup_template.read()
            template_source = f"uploaded:{mockup_template.filename}"
            print(f"[Render] Using uploaded template {mockup_template.filename}")
        else:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "success": False,
                    "errorType": "RENDER_ERROR",
                    "message": f"No template found for category='{category}' mockup_index='{mockup_index}' and no mockup_template was uploaded."
                }
            )

        # ── Read product images ──────────────────────────────────────────────
        product_bytes_list = []
        for img in [image_1, image_2, image_3, image_4, image_5, image_6]:
            if img is not None:
                product_bytes_list.append(await img.read())

        if not product_bytes_list:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "errorType": "RENDER_ERROR",
                    "message": "At least one product image is required."
                }
            )

        # ── Coordinate lookup ────────────────────────────────────────────────
        frames, calibrated, settings = lookup_frames(category, mockup_index)
        print(f"[Render] coordinates={category}/{mockup_index} calibrated={str(calibrated).lower()}")

        # ── Render ───────────────────────────────────────────────────────────
        print(f"[Render] method=opencv-local")
        rendered_bytes = warp_perspective_cv(
            template_bytes=base_bytes,
            product_bytes_list=product_bytes_list,
            frames=frames,
            settings=settings,
            mask_bytes=mask_bytes,
            overlay_bytes=overlay_bytes
        )

        duration_ms = int((time.time() - start_time) * 1000)
        print(f"[Render] complete bytes={len(rendered_bytes)} time={duration_ms}ms")

        return Response(
            content=rendered_bytes,
            media_type="image/jpeg",
            headers={
                "X-Render-Method": "opencv-local",
                "X-Render-Time-Ms": str(duration_ms),
                "X-Calibrated": str(calibrated).lower(),
                "Content-Length": str(len(rendered_bytes)),
                "Content-Disposition": f'inline; filename="rendered_mockup_{mockup_index}.jpg"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[Render ERROR] category='{category}' index='{mockup_index}': {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "errorType": "RENDER_ERROR",
                "message": f"Rendering pipeline failed: {str(e)}"
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
