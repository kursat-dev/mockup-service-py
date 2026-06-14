import cv2
import numpy as np
import os
import json

def detect_corners(img_path, expected_count=1):
    img = cv2.imread(img_path)
    if img is None:
        return None, "Failed to load image file"
    
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try Canny first
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_quads = []
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(c)
            if area > (w * h * 0.015): # at least 1.5% of the image
                x, y, bw, bh = cv2.boundingRect(approx)
                aspect_ratio = float(bw) / bh
                if 0.35 < aspect_ratio < 2.5:
                    detected_quads.append((area, approx))
                    
    detected_quads.sort(key=lambda x: x[0], reverse=True)
    
    # Fallback to thresholding if Canny is too strict
    if len(detected_quads) < expected_count:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        detected_quads = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(c)
                if area > (w * h * 0.015):
                    x, y, bw, bh = cv2.boundingRect(approx)
                    aspect_ratio = float(bw) / bh
                    if 0.35 < aspect_ratio < 2.5:
                        detected_quads.append((area, approx))
        detected_quads.sort(key=lambda x: x[0], reverse=True)
        
    selected_quads = detected_quads[:expected_count]
    
    if len(selected_quads) < expected_count:
        return None, f"Detected {len(selected_quads)} frames, expected {expected_count}"
        
    quads_with_centroids = []
    for area, approx in selected_quads:
        pts = approx.reshape(4, 2)
        centroid_x = np.mean(pts[:, 0])
        centroid_y = np.mean(pts[:, 1])
        quads_with_centroids.append((centroid_x, centroid_y, pts))
        
    # Grid sort logic
    if expected_count == 6:
        # Sort by Y first
        quads_with_centroids.sort(key=lambda q: q[1])
        # Divide into row 1 and row 2
        row1 = sorted(quads_with_centroids[:3], key=lambda q: q[0])
        row2 = sorted(quads_with_centroids[3:], key=lambda q: q[0])
        sorted_quads = row1 + row2
    else:
        # Single or triple sort by X
        sorted_quads = sorted(quads_with_centroids, key=lambda q: q[0])
        
    result_frames = []
    for idx, (cx, cy, pts) in enumerate(sorted_quads):
        # Clockwise sort TL, TR, BR, BL
        # Sum (x+y) gives TL (min) and BR (max)
        # Diff (y-x) gives TR (min) and BL (max)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        
        corners_norm = {
            "top_left": [int(round((tl[0] / w) * 1000)), int(round((tl[1] / h) * 1000))],
            "top_right": [int(round((tr[0] / w) * 1000)), int(round((tr[1] / h) * 1000))],
            "bottom_right": [int(round((br[0] / w) * 1000)), int(round((br[1] / h) * 1000))],
            "bottom_left": [int(round((bl[0] / w) * 1000)), int(round((bl[1] / h) * 1000))]
        }
        
        result_frames.append({
            "index": idx + 1,
            "corners": corners_norm
        })
        
    return result_frames, "Success"

def main():
    base_dir = "/Users/kursat/Documents/formockup"
    folders = {
        "tekliler": (1, "single"),
        "kitchen": (1, "kitchen"),
        "3lüler": (3, "triple"),
        "6lılar": (6, "six")
    }
    
    database = {}
    low_confidence = []
    
    print("=== Automated Frame Coordinate Extraction (OpenCV) ===")
    for folder, (expected, category) in folders.items():
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Skipping directory {folder} (not found)")
            continue
            
        print(f"\nAnalyzing '{folder}' directory...")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        files.sort()
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            # Both direct simple name and key name
            simple_name = file
            filename_key = f"{category}/{file}"
            
            frames, msg = detect_corners(file_path, expected)
            if frames:
                database[simple_name] = frames
                database[filename_key] = frames
                print(f"  ✅ SUCCESS: '{file}' -> Detected {len(frames)} frame(s)")
            else:
                low_confidence.append({
                    "file": file,
                    "category": category,
                    "reason": msg,
                    "path": file_path
                })
                print(f"  ⚠️  LOW CONFIDENCE: '{file}' -> {msg}")
                
    # Write output to coordinates.json
    db_out_path = os.path.join(base_dir, "mockup-service-py", "app", "database", "coordinates.json")
    # Load existing coordinates to merge/update
    existing = {}
    if os.path.exists(db_out_path):
        try:
            with open(db_out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
            
    existing.update(database)
    
    with open(db_out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        
    print(f"\n=== Detection Completed ===")
    print(f"Successfully detected and wrote: {len(database)//2} templates")
    print(f"Low confidence templates: {len(low_confidence)}")
    for item in low_confidence:
        print(f"  - [{item['category'].upper()}] '{item['file']}': {item['reason']}")

if __name__ == "__main__":
    main()
