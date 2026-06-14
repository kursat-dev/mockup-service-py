import json
import os
import shutil
import cv2
import numpy as np

def main():
    db_path = "/Users/kursat/Documents/formockup/mockup-service-py/app/database/coordinates.json"
    templates_dir = "/Users/kursat/Documents/formockup/mockup-service-py/templates"
    
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
        
    quality_settings = {
        "feather_px": 1.5,
        "supersample": 4,
        "brightness_match": 0.2,
        "saturation_match": 0.1,
        "inner_shadow": True,
        "overlay_opacity": 1.0,
        "jpeg_quality": 95
    }
    
    for i in range(4, 10):
        idx_str = str(i)
        print(f"Migrating single-{idx_str}...")
        
        # 1. Update coordinates database
        entry = db["single"][idx_str]
        for k, v in quality_settings.items():
            entry[k] = v
            
        # 2. Create directory
        tpl_dir = os.path.join(templates_dir, "single", f"single-{idx_str}")
        os.makedirs(tpl_dir, exist_ok=True)
        
        # 3. Copy legacy file to base.jpg
        legacy_file = os.path.join(templates_dir, "single", f"single-{idx_str}.jpeg")
        base_path = os.path.join(tpl_dir, "base.jpg")
        shutil.copy(legacy_file, base_path)
        
        # 4. Generate mask.png from coordinates with antialiased edges
        base_img = cv2.imread(base_path)
        h, w = base_img.shape[:2]
        
        frames = entry.get("frames", [])
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
        else:
            mask_img = np.zeros((h, w), dtype=np.uint8)
            
        cv2.imwrite(os.path.join(tpl_dir, "mask.png"), mask_img)
        
        # 5. Create transparent overlay.png
        overlay_img = np.zeros((h, w, 4), dtype=np.uint8)
        cv2.imwrite(os.path.join(tpl_dir, "overlay.png"), overlay_img)
        
        print(f"Successfully migrated single-{idx_str}")
        
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
        
    print("Coordinates DB saved successfully!")

if __name__ == "__main__":
    main()
