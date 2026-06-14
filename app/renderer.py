import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional

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


# ── Main renderer ─────────────────────────────────────────────────────────────

def warp_perspective_cv(
    template_bytes: bytes,
    product_bytes_list: list,
    frames: list,
    settings: Optional[dict] = None,
    mask_bytes: Optional[bytes] = None,
    overlay_bytes: Optional[bytes] = None,
    # Old params kept for backward compatibility:
    inner_shadow: bool = True,
    glass_glare: bool = False,
    paper_texture: bool = False,
    lighting_multiply: bool = True
) -> bytes:
    """
    Deterministic Computer-Vision Rendering Pipeline
    Supports supersampling, Lanczos downsampling, feathered mask,
    color/luminance matching, and base/mask/overlay template composition.
    """
    # ── 1. Initialize settings ──
    if settings is None:
        settings = {}
    
    # Safe defaults
    feather_px = settings.get("feather_px", 1.5)
    supersample = int(settings.get("supersample", 4))
    brightness_match = settings.get("brightness_match", 0.2)
    saturation_match = settings.get("saturation_match", 0.1)
    use_inner_shadow = settings.get("inner_shadow", inner_shadow)
    overlay_opacity = settings.get("overlay_opacity", 1.0)
    jpeg_quality = int(settings.get("jpeg_quality", 95))

    # ── 2. Load base image ──
    arr_base = np.frombuffer(template_bytes, np.uint8)
    img_base = cv2.imdecode(arr_base, cv2.IMREAD_COLOR)   # BGR
    if img_base is None:
        raise ValueError("Failed to decode mockup template/base image.")
    th, tw = img_base.shape[:2]

    # ── 3. Load global mask ──
    img_mask = None
    if mask_bytes is not None:
        arr_mask = np.frombuffer(mask_bytes, np.uint8)
        img_mask = cv2.imdecode(arr_mask, cv2.IMREAD_GRAYSCALE)

    # ── 4. Load overlay ──
    img_overlay = None
    if overlay_bytes is not None:
        arr_overlay = np.frombuffer(overlay_bytes, np.uint8)
        img_overlay = cv2.imdecode(arr_overlay, cv2.IMREAD_UNCHANGED)  # Load BGRA

    # ── 5. Render each frame ──
    img_composite = img_base.copy()

    for i, frame in enumerate(frames):
        if i >= len(product_bytes_list):
            print(f"[Renderer] Frame {i+1} skipped — no matching product image supplied.")
            break

        arr_prod = np.frombuffer(product_bytes_list[i], np.uint8)
        img_prod = cv2.imdecode(arr_prod, cv2.IMREAD_COLOR)  # BGR
        if img_prod is None:
            print(f"[Renderer] Frame {i+1} skipped — could not decode product image.")
            continue
        ph, pw = img_prod.shape[:2]

        corners = frame["corners"]

        # Destination points in template space
        pts_dst = np.array([
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"]
        ], dtype=np.float32)

        # ── Color/Luminance Matching ──
        # Estimate from target frame area on base image
        frame_poly = pts_dst.astype(np.int32)
        temp_mask = np.zeros((th, tw), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [frame_poly], 255)
        
        mean_color = cv2.mean(img_base, mask=temp_mask)[:3]  # BGR
        b_base, g_base, r_base = mean_color
        
        # Calculate luminance (standard coefficients)
        L_base = 0.299 * r_base + 0.587 * g_base + 0.114 * b_base
        
        if L_base > 0:
            t_r = r_base / L_base
            t_g = g_base / L_base
            t_b = b_base / L_base
        else:
            t_r, t_g, t_b = 1.0, 1.0, 1.0

        # Adjust brightness gently
        m_bright = L_base / 255.0
        m_bright_adj = 1.0 + (m_bright - 1.0) * brightness_match
        
        # Adjust saturation/temperature gently
        t_r_adj = 1.0 + (t_r - 1.0) * saturation_match
        t_g_adj = 1.0 + (t_g - 1.0) * saturation_match
        t_b_adj = 1.0 + (t_b - 1.0) * saturation_match
        
        factor_r = t_r_adj * m_bright_adj
        factor_g = t_g_adj * m_bright_adj
        factor_b = t_b_adj * m_bright_adj
        
        # Apply BGR adjustments to artwork
        img_matched = img_prod.astype(float)
        img_matched[:, :, 0] = np.clip(img_matched[:, :, 0] * factor_b, 0.0, 255.0)
        img_matched[:, :, 1] = np.clip(img_matched[:, :, 1] * factor_g, 0.0, 255.0)
        img_matched[:, :, 2] = np.clip(img_matched[:, :, 2] * factor_r, 0.0, 255.0)
        img_matched = img_matched.astype(np.uint8)

        # ── Inner Contact Shadow ──
        if use_inner_shadow:
            prod_pil = Image.fromarray(cv2.cvtColor(img_matched, cv2.COLOR_BGR2RGB)).convert("RGBA")
            shadow = _build_inner_shadow(pw, ph)
            prod_pil = Image.alpha_composite(prod_pil, shadow)
            img_matched = cv2.cvtColor(np.array(prod_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        # ── Supersampled Warp ──
        sw, sh = tw * supersample, th * supersample
        pts_dst_scaled = pts_dst * supersample
        
        pts_src = np.array([
            [0, 0],
            [pw - 1, 0],
            [pw - 1, ph - 1],
            [0, ph - 1]
        ], dtype=np.float32)

        h_matrix_scaled = cv2.getPerspectiveTransform(pts_src, pts_dst_scaled)
        
        # Warp artwork at Nx resolution
        warped_art_scaled = cv2.warpPerspective(img_matched, h_matrix_scaled, (sw, sh), flags=cv2.INTER_LANCZOS4)
        
        # Warp mask at Nx resolution
        mask_src = np.ones((ph, pw), dtype=np.uint8) * 255
        warped_mask_scaled = cv2.warpPerspective(mask_src, h_matrix_scaled, (sw, sh), flags=cv2.INTER_NEAREST)

        # ── Downsampling ──
        warped_art = cv2.resize(warped_art_scaled, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        warped_mask = cv2.resize(warped_mask_scaled, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        # ── Feathering the Mask ──
        if feather_px > 0:
            ksize = int(2 * round(3 * feather_px) + 1)
            ksize = max(3, ksize | 1)
            warped_mask_feathered = cv2.GaussianBlur(warped_mask, (ksize, ksize), feather_px)
        else:
            warped_mask_feathered = warped_mask.copy()

        # Combine with global mask if available
        if img_mask is not None:
            m_feathered = warped_mask_feathered.astype(float) / 255.0
            m_global = img_mask.astype(float) / 255.0
            final_mask = (m_feathered * m_global * 255.0).astype(np.uint8)
        else:
            final_mask = warped_mask_feathered

        # ── Blend warped artwork onto base ──
        mask_normalized = final_mask.astype(float) / 255.0
        mask_3d = np.expand_dims(mask_normalized, axis=2)
        
        img_composite = (warped_art.astype(float) * mask_3d + img_composite.astype(float) * (1.0 - mask_3d)).astype(np.uint8)

    # ── 6. Composite overlay.png ──
    if img_overlay is not None:
        overlay_bgr = img_overlay[:, :, :3]
        overlay_alpha = (img_overlay[:, :, 3].astype(float) / 255.0) * overlay_opacity
        overlay_alpha_3d = np.expand_dims(overlay_alpha, axis=2)
        
        img_composite = (overlay_bgr.astype(float) * overlay_alpha_3d + img_composite.astype(float) * (1.0 - overlay_alpha_3d)).astype(np.uint8)

    # ── 7. Encode as JPEG ──
    success, encoded = cv2.imencode(".jpg", img_composite, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not success:
        raise ValueError("Failed to encode composite image as JPEG.")
    return encoded.tobytes()

