import cv2
import numpy as np
from PIL import Image, ImageChops
import io


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_inner_shadow(pw: int, ph: int, depth_ratio: float = 0.035, max_opacity: int = 90) -> Image.Image:
    """
    Generates a soft inner bezel shadow that darkens only the border region.
    depth_ratio : fraction of shortest dimension used as the shadow depth.
    max_opacity : maximum alpha at the very edge (0–255).
    """
    y_idx, x_idx = np.indices((ph, pw))
    dist_left   = x_idx
    dist_right  = (pw - 1) - x_idx
    dist_top    = y_idx
    dist_bottom = (ph - 1) - y_idx
    min_dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))

    depth = max(2.0, min(pw, ph) * depth_ratio)
    factor = np.clip(min_dist / depth, 0.0, 1.0)
    alpha  = (max_opacity * (1.0 - np.power(factor, 0.55))).astype(np.uint8)

    shadow_np = np.zeros((ph, pw, 4), dtype=np.uint8)
    shadow_np[:, :, 3] = alpha
    return Image.fromarray(shadow_np, "RGBA")


def _extract_luminance_map(
    img_temp_bgr: np.ndarray,
    h_matrix_inv: np.ndarray,
    pw: int,
    ph: int,
    baseline: float = 230.0
) -> np.ndarray:
    """
    Back-warps the template patch into artwork space, extracts a smooth
    luminance map, and returns it normalised around a mid-tone baseline.
    Values <1.0 darken the artwork; values >1.0 brighten it.
    """
    patch = cv2.warpPerspective(img_temp_bgr, h_matrix_inv, (pw, ph), flags=cv2.INTER_LINEAR)
    gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(float)

    # Keep only low-frequency (large-scale) lighting gradients
    ksize = max(5, int(min(pw, ph) * 0.12) | 1)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Clamp range so highlights don't wash out and shadows don't go black
    lum = np.clip(blurred / baseline, 0.10, 1.15)
    return lum


# ── Main renderer ─────────────────────────────────────────────────────────────

def warp_perspective_cv(
    template_bytes: bytes,
    product_bytes_list: list,
    frames: list,
    inner_shadow: bool = True,
    glass_glare: bool = False,
    paper_texture: bool = False,
    lighting_multiply: bool = True
) -> bytes:
    """
    Deterministic Computer-Vision Rendering Pipeline
    
    Parameters
    ----------
    template_bytes      : raw bytes of the JPEG mockup background
    product_bytes_list  : list of raw bytes, one per artwork (matching frame order)
    frames              : list of frame dicts with {'index', 'corners': {top_left, …}}
                          Corners are PIXEL coordinates in the template image space.
    inner_shadow        : soft bezel shadow at artwork edges (default True)
    glass_glare         : diagonal specular highlight (default False — too artificial)
    paper_texture       : random grain overlay (default False — tends to smudge)
    lighting_multiply   : multiply artwork by local template luminance (default True)
    
    Returns
    -------
    bytes : JPEG-encoded composite image
    """
    # ── 1. Load template ─────────────────────────────────────────────────────
    arr_temp = np.frombuffer(template_bytes, np.uint8)
    img_temp = cv2.imdecode(arr_temp, cv2.IMREAD_COLOR)   # BGR
    if img_temp is None:
        raise ValueError("Failed to decode mockup template image.")
    th, tw = img_temp.shape[:2]

    pil_template = Image.fromarray(cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # ── 2. Render each frame ─────────────────────────────────────────────────
    for i, frame in enumerate(frames):
        if i >= len(product_bytes_list):
            print(f"[Renderer] Frame {i+1} skipped — no matching product image supplied.")
            break

        arr_prod = np.frombuffer(product_bytes_list[i], np.uint8)
        img_prod = cv2.imdecode(arr_prod, cv2.IMREAD_COLOR)
        if img_prod is None:
            print(f"[Renderer] Frame {i+1} skipped — could not decode product image.")
            continue
        ph, pw = img_prod.shape[:2]

        corners = frame["corners"]

        # Destination points — already in PIXEL space of the template
        pts_dst = np.array([
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"]
        ], dtype=np.float32)

        # Source = full product image corners
        pts_src = np.array([
            [0,      0],
            [pw - 1, 0],
            [pw - 1, ph - 1],
            [0,      ph - 1]
        ], dtype=np.float32)

        h_matrix     = cv2.getPerspectiveTransform(pts_src, pts_dst)
        h_matrix_inv = cv2.getPerspectiveTransform(pts_dst, pts_src)

        # Convert product to Pillow RGBA for effect compositing
        prod_pil = Image.fromarray(cv2.cvtColor(img_prod, cv2.COLOR_BGR2RGB)).convert("RGBA")

        # ── Effect A: Local Lighting Multiply ────────────────────────────────
        if lighting_multiply:
            lum = _extract_luminance_map(img_temp, h_matrix_inv, pw, ph)
            prod_np = np.array(prod_pil.convert("RGB")).astype(float)
            for ch in range(3):
                prod_np[:, :, ch] = np.clip(prod_np[:, :, ch] * lum, 0.0, 255.0)
            prod_pil = Image.fromarray(prod_np.astype(np.uint8)).convert("RGBA")

        # ── Effect B: Inner Bezel Shadow ─────────────────────────────────────
        if inner_shadow:
            shadow = _build_inner_shadow(pw, ph)
            prod_pil = Image.alpha_composite(prod_pil, shadow)

        # ── Effect C: Glass Glare (disabled by default) ──────────────────────
        if glass_glare:
            glare_np = np.zeros((ph, pw, 4), dtype=np.uint8)
            y_idx, x_idx = np.indices((ph, pw))
            u = x_idx / max(1.0, float(pw - 1))
            v = y_idx / max(1.0, float(ph - 1))
            diag = u + v
            glare = np.zeros_like(diag)
            m1 = (diag >= 0.70) & (diag <= 1.25)
            glare[m1] = np.power(1.0 - np.abs(diag[m1] - 0.95) / 0.28, 3.5) * 22
            m2 = (diag >= 0.18) & (diag <= 0.38)
            glare[m2] = np.power(1.0 - np.abs(diag[m2] - 0.28) / 0.10, 2) * 5
            glare_np[:, :, :3] = 255
            glare_np[:, :,  3] = glare.astype(np.uint8)
            prod_pil = Image.alpha_composite(prod_pil, Image.fromarray(glare_np, "RGBA"))

        # ── Warp into template space ─────────────────────────────────────────
        final_bgr = cv2.cvtColor(np.array(prod_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        warped_art  = cv2.warpPerspective(final_bgr, h_matrix, (tw, th), flags=cv2.INTER_LANCZOS4)
        mask_src    = np.ones((ph, pw), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask_src, h_matrix, (tw, th), flags=cv2.INTER_NEAREST)

        # Composite onto master template
        art_rgb = cv2.cvtColor(warped_art, cv2.COLOR_BGR2RGB)
        pil_art = Image.fromarray(art_rgb).convert("RGBA")
        pil_art.putalpha(Image.fromarray(warped_mask).convert("L"))
        pil_template.alpha_composite(pil_art)

    # ── 3. Encode as JPEG ────────────────────────────────────────────────────
    buf = io.BytesIO()
    pil_template.convert("RGB").save(buf, format="JPEG", quality=93, optimize=True)
    return buf.getvalue()
