"""
Precise Color + Shape + Object Detector  (single window)

Install once:
    pip install opencv-python numpy ultralytics

How to use:
    * Click the tabs at the top (NORMAL / BLACK SCREEN) - or press TAB, V, 1, 2.
    * Press Q (or ESC) to quit.

What it detects:
    * Real objects (bottle, scissors, cup, phone, ...) using a YOLOv8
      *segmentation* model, so you get the object's true outline (not just a
      box) and its color is measured only on the object itself.
    * Colored shapes -> only Circle, Rectangle or Triangle (e.g. "Red Circle")
"""
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
WINDOW = "Precise Detector"
DISP_W, TAB_H = 960, 44
START_WINDOW_WIDTH = 900   # starting window width in pixels (change this if it's too big/small)

# Fixed detection settings (edit these numbers here if you ever want to tune things)
FIND_OBJECTS = True        # bottles, scissors, cups, phones, ...
FIND_SHAPES = True         # colored circles / rectangles / triangles
OBJECT_CONFIDENCE = 0.50   # higher = fewer false object detections
COLOR_STRICTNESS = 80      # min color saturation (higher = only vivid colors)
IGNORE_DARK_BELOW = 50     # min brightness (higher = skip darker pixels)
HUE_WIDEN = 0              # 0-8, raise to accept more shades of each color
MIN_SHAPE_AREA = 800       # ignore shapes smaller than this many pixels
SHAPE_STRICTNESS = 3       # 1-10, higher = shape must match more exactly
LIGHTING_FIX = False       # True evens out shadows and glare

COLORS_BGR = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
}
NEUTRAL_BGR = (255, 255, 255)
HUE_RANGES = {
    "Red": [(0, 8), (172, 179)],
    "Yellow": [(20, 35)],
    "Green": [(40, 85)],
    "Blue": [(95, 130)],
}

MIN_SOLIDITY = 0.75
MAX_AREA_FRACTION = 0.60
OBJECT_COLOR_MIN_FRACTION = 0.15   # share of the object's own pixels needed to name a color

TRACK_MATCH_DIST = 80
TRACK_MAX_MISSED = 10
TRACK_DRAW_MISSED = 3
HISTORY_LEN = 15
LOCK_MIN_FRAMES = 3
LOCK_MIN_RATIO = 0.60
CENTER_SMOOTHING = 0.6


# ----------------------------------------------------------------------------
# Color masks
# ----------------------------------------------------------------------------
def build_limits(hue_widen, s_min, v_min):
    limits = {}
    for name, ranges in HUE_RANGES.items():
        limits[name] = [
            (
                np.array((max(0, lo - hue_widen), s_min, v_min), dtype=np.uint8),
                np.array((min(179, hi + hue_widen), 255, 255), dtype=np.uint8),
            )
            for lo, hi in ranges
        ]
    return limits


# ----------------------------------------------------------------------------
# Geometric shape classification (for plain colored blobs)
# ----------------------------------------------------------------------------
def _angle(a, b, c):
    v1, v2 = a - b, c - b
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if d == 0:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / d, -1.0, 1.0))))


def classify_shape(contour, strictness=5):
    """Return (shape, outline) where shape is only Circle / Rectangle / Triangle.

    Decision uses the measured width and height of the tightest rotated box
    around the shape (works at any rotation):
        extent   = shape area / (width * height)   rectangle ~1.0, circle ~0.79, triangle ~0.5
        aspect   = short side / long side          circle ~1.0
        tri_fill = shape area / smallest enclosing triangle area   triangle ~1.0
        circ_fill= shape area / smallest enclosing circle area     circle ~1.0
    `strictness` (1-10) tightens all the thresholds together.
    """
    t = np.clip(strictness, 1, 10) / 10.0
    rect_extent_min = 0.80 + 0.12 * t       # 0.81 .. 0.92
    tri_fill_min = 0.80 + 0.14 * t          # 0.81 .. 0.94
    circ_circularity_min = 0.76 + 0.14 * t  # 0.77 .. 0.90
    circ_fill_min = 0.74 + 0.14 * t         # 0.75 .. 0.88
    circ_aspect_min = 0.75 + 0.15 * t       # 0.76 .. 0.90

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(contour)
    if hull_area <= 0 or area <= 0 or area / hull_area < MIN_SOLIDITY:
        return None, None
    peri = cv2.arcLength(hull, True)
    if peri == 0:
        return None, None

    (rcx, rcy), (w, h), _ = cv2.minAreaRect(hull)
    if w <= 0 or h <= 0:
        return None, None
    extent = area / (w * h)
    aspect = min(w, h) / max(w, h)
    circularity = 4 * np.pi * hull_area / (peri * peri)

    # 1) Rectangle: fills its rotated bounding box almost completely
    if extent >= rect_extent_min:
        box = cv2.boxPoints(cv2.minAreaRect(hull))
        return "Rectangle", np.round(box).astype(np.int32).reshape(-1, 1, 2)

    # 2) Circle: width ~= height, round, and fills its enclosing circle
    (ccx, ccy), radius = cv2.minEnclosingCircle(hull)
    circ_fill = area / (np.pi * radius * radius) if radius > 0 else 0
    # A real circle has no distinct corners; a hexagon/octagon does (few polygon vertices).
    n_corners = len(cv2.approxPolyDP(hull, 0.01 * peri, True))
    if (aspect >= circ_aspect_min and circularity >= circ_circularity_min
            and circ_fill >= circ_fill_min and n_corners >= 8):
        pts = cv2.ellipse2Poly((int(ccx), int(ccy)), (int(radius), int(radius)), 0, 0, 360, 6)
        return "Circle", pts.reshape(-1, 1, 2).astype(np.int32)

    # 3) Triangle: fills its smallest enclosing triangle
    tri_area, tri = cv2.minEnclosingTriangle(hull)
    if tri is not None and tri_area > 0 and area / tri_area >= tri_fill_min:
        return "Triangle", np.round(tri).astype(np.int32).reshape(-1, 1, 2)

    # 4) Fallback for tilted / blurry real-world shapes: count the corners instead
    approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
    n = len(approx)
    if n == 3:
        return "Triangle", approx
    if n == 4:
        p = approx.reshape(-1, 2).astype(np.float64)
        angles = [_angle(p[i - 1], p[i], p[(i + 1) % 4]) for i in range(4)]
        if all(60 <= a <= 120 for a in angles):     # allows perspective skew
            return "Rectangle", approx
    if n >= 7 and aspect >= 0.70 and circularity >= 0.75:
        pts = cv2.ellipse2Poly((int(ccx), int(ccy)), (int(radius), int(radius)), 0, 0, 360, 6)
        return "Circle", pts.reshape(-1, 1, 2).astype(np.int32)

    return None, None


# ----------------------------------------------------------------------------
# Object recognition on a background thread (keeps the UI smooth)
# ----------------------------------------------------------------------------
class ObjectDetector:
    MODELS = {
        "Fast": ("yolov8n-seg.pt", 640),
        "Balanced": ("yolov8s-seg.pt", 800),
        "Precise": ("yolov8m-seg.pt", 960),
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._frame, self._fid = None, 0
        self._settings = ("Balanced", 0.5)
        self._results = []
        self._stop = False
        self.status = "starting..."
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame, mode, conf):
        with self._lock:
            self._frame, self._settings = frame, (mode, conf)
            self._fid += 1

    def results(self):
        with self._lock:
            return list(self._results)

    def clear(self):
        with self._lock:
            self._results = []
            self._frame = None

    def stop(self):
        self._stop = True

    def _run(self):
        model, loaded, last = None, None, -1
        while not self._stop:
            with self._lock:
                frame, fid, (mode, conf) = self._frame, self._fid, self._settings
            if frame is None or fid == last:
                time.sleep(0.01)
                continue
            last = fid
            try:
                name, imgsz = self.MODELS[mode]
                if loaded != name:
                    self.status = f"loading {name} (first run downloads it)..."
                    model = YOLO(name)
                    loaded = name
                res = model(
                    frame, conf=conf, imgsz=imgsz, iou=0.5,
                    agnostic_nms=True, verbose=False,
                )[0]
                names = res.names
                polys = res.masks.xy if res.masks is not None else None
                out = []
                for i, b in enumerate(res.boxes):
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    poly = None
                    if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                        poly = np.round(polys[i]).astype(np.int32).reshape(-1, 1, 2)
                    out.append({
                        "name": names[int(b.cls[0])].title(),
                        "conf": float(b.conf[0]),
                        "box": (x1, y1, x2, y2),
                        "poly": poly,
                    })
                with self._lock:
                    self._results = out
                self.status = f"ready - {name}"
            except Exception as e:  # noqa: BLE001
                self.status = f"ERROR: {str(e)[:70]}"
                time.sleep(1.0)


def default_accuracy_index():
    """Pick 'Precise' automatically when a GPU is available, else 'Balanced'."""
    try:
        import torch
        if torch.cuda.is_available():
            return 2
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return 2
    except Exception:  # noqa: BLE001
        pass
    return 1


# ----------------------------------------------------------------------------
# Tracking (keeps labels stable between frames)
# ----------------------------------------------------------------------------
class Track:
    def __init__(self, tid, key, center):
        self.id, self.key = tid, key
        self.center = np.array(center, dtype=np.float64)
        self.approx, self.bgr, self.conf = None, NEUTRAL_BGR, None
        self.history = deque(maxlen=HISTORY_LEN)
        self.locked_label = None
        self.missed = 0

    def update(self, det):
        self.center = (CENTER_SMOOTHING * np.array(det["center"], dtype=np.float64)
                       + (1.0 - CENTER_SMOOTHING) * self.center)
        self.approx, self.bgr, self.conf = det["approx"], det["bgr"], det.get("conf")
        self.missed = 0
        self.history.append(det["label"])
        if len(self.history) >= LOCK_MIN_FRAMES:
            top, count = Counter(self.history).most_common(1)[0]
            if count >= LOCK_MIN_RATIO * len(self.history):
                self.locked_label = top


def assign_detections(tracks, detections):
    pairs = []
    for di, det in enumerate(detections):
        for tid, trk in tracks.items():
            if trk.key != det["key"]:
                continue
            dist = float(np.hypot(*(trk.center - np.array(det["center"]))))
            if dist < TRACK_MATCH_DIST:
                pairs.append((dist, tid, di))
    pairs.sort()
    used_t, used_d, matches = set(), set(), []
    for _, tid, di in pairs:
        if tid in used_t or di in used_d:
            continue
        used_t.add(tid)
        used_d.add(di)
        matches.append((tid, di))
    return matches, used_d


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def box_to_poly(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32).reshape(-1, 1, 2)


def dominant_color_in_poly(masks, poly, W, H):
    """Which of our 4 colors covers the most of the object's OWN pixels."""
    x, y, w, h = cv2.boundingRect(poly)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    local = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(local, [poly - np.array([x1, y1], dtype=np.int32)], 255)
    eroded = cv2.erode(local, np.ones((5, 5), np.uint8))   # ignore blurry edge pixels
    if cv2.countNonZero(eroded) > 50:
        local = eroded
    area = float(cv2.countNonZero(local))
    if area == 0:
        return None
    best, best_frac = None, 0.0
    for name, mask in masks.items():
        frac = cv2.countNonZero(cv2.bitwise_and(mask[y1:y2, x1:x2], local)) / area
        if frac > best_frac:
            best, best_frac = name, frac
    return best if best_frac >= OBJECT_COLOR_MIN_FRACTION else None


def draw_label(img, text, org, color, scale=0.85):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 6), (x + tw + 3, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_tracks(canvas, tracks):
    active = []
    for trk in tracks.values():
        if trk.locked_label is None or trk.missed > TRACK_DRAW_MISSED or trk.approx is None:
            continue
        active.append(trk.locked_label)
        cv2.drawContours(canvas, [trk.approx], -1, trk.bgr, 3)
        text = trk.locked_label
        if trk.conf is not None:
            text += f" {trk.conf * 100:.0f}%"
        bx, by, _, _ = cv2.boundingRect(trk.approx)
        draw_label(canvas, text, (max(5, bx), max(28, by - 10)), trk.bgr)
    return active


def put_outlined(img, text, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


# ----------------------------------------------------------------------------
# Tab bar UI
# ----------------------------------------------------------------------------
def inside(rect, x, y):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


class Layout:
    def __init__(self, disp_h):
        self.disp_h = disp_h
        self.tabs = [(8, 6, 188, 38), (196, 6, 396, 38)]


def build_ui(video, view, layout, status_lines):
    disp = cv2.resize(video, (DISP_W, layout.disp_h), interpolation=cv2.INTER_AREA)
    for i, line in enumerate(status_lines):
        put_outlined(disp, line, (10, layout.disp_h - 12 - 22 * (len(status_lines) - 1 - i)))

    ui = np.full((TAB_H + layout.disp_h, DISP_W, 3), 32, np.uint8)
    ui[TAB_H:, :DISP_W] = disp

    for idx, (name, rect) in enumerate(zip(["NORMAL", "BLACK SCREEN"], layout.tabs)):
        active = idx == view
        cv2.rectangle(ui, rect[:2], rect[2:], (60, 170, 60) if active else (75, 75, 75), -1)
        cv2.rectangle(ui, rect[:2], rect[2:], (255, 255, 255) if active else (130, 130, 130), 1)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cx = (rect[0] + rect[2]) // 2 - tw // 2
        cv2.putText(ui, name, (cx, rect[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(ui, "or click the picture / press SPACE", (410, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return ui


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("Starting Precise Detector...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open webcam! Check permissions or close other apps.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        return
    H, W = frame.shape[:2]
    disp_h = int(DISP_W * H / W)

    accuracy = ['Fast', 'Balanced', 'Precise'][default_accuracy_index()]
    layout = Layout(disp_h)
    state = {"view": 0}

    def on_mouse(event, x, y, flags, param):
        # Right-click anywhere = switch view
        if event == cv2.EVENT_RBUTTONDOWN:
            state["view"] = 1 - state["view"]
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        hit_tab = False
        for idx, rect in enumerate(layout.tabs):
            if inside(rect, x, y):
                state["view"] = idx
                hit_tab = True
        # Left-click on the video picture itself = switch view
        if not hit_tab and x < DISP_W and TAB_H <= y < TAB_H + layout.disp_h:
            state["view"] = 1 - state["view"]
            return

    # Resizable window: drag the edges / maximize button, or press [ and ] to shrink / grow
    ui_w = DISP_W
    ui_h = TAB_H + disp_h
    win_w = START_WINDOW_WIDTH
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, win_w, int(win_w * ui_h / ui_w))
    cv2.setMouseCallback(WINDOW, on_mouse)

    detector = ObjectDetector() if YOLO is not None else None
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    tracks, next_id = {}, 0
    last_announced = set()
    fps, t_prev = 0.0, time.time()

    print("Running! Click the tabs (or press SPACE) to switch views. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # ---- read settings -------------------------------------------------
        conf = OBJECT_CONFIDENCE
        objects_on = FIND_OBJECTS
        shapes_on = FIND_SHAPES
        s_min = COLOR_STRICTNESS
        v_min = IGNORE_DARK_BELOW
        hue_widen = HUE_WIDEN
        min_area = MIN_SHAPE_AREA
        shape_strict = SHAPE_STRICTNESS
        light_fix = LIGHTING_FIX
        max_area = MAX_AREA_FRACTION * H * W

        # ---- color masks ---------------------------------------------------
        smooth = cv2.bilateralFilter(frame, 5, 50, 50)
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
        if light_fix:
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

        black_screen = np.zeros_like(frame)
        masks = {}
        for name, rngs in build_limits(hue_widen, s_min, v_min).items():
            m = None
            for lo, hi in rngs:
                part = cv2.inRange(hsv, lo, hi)
                m = part if m is None else cv2.bitwise_or(m, part)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
            masks[name] = m
            black_screen[m > 0] = COLORS_BGR[name]

        detections = []

        # ---- objects (from the background thread) ---------------------------
        object_polys = []
        if detector is not None and objects_on:
            detector.submit(frame, accuracy, conf)
            for r in detector.results():
                poly = r["poly"] if r["poly"] is not None else box_to_poly(*r["box"])
                col = dominant_color_in_poly(masks, poly, W, H)
                label = f"{col} {r['name']}" if col else r["name"]
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                else:
                    x1, y1, x2, y2 = r["box"]
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                detections.append({
                    "key": f"obj:{r['name']}", "label": label,
                    "bgr": COLORS_BGR.get(col, NEUTRAL_BGR),
                    "approx": poly, "center": center,
                    "area": float(cv2.contourArea(poly)), "conf": r["conf"],
                })
                object_polys.append(poly)
        elif detector is not None:
            detector.clear()

        # ---- plain colored shapes ------------------------------------------
        blobs_seen = 0
        if shapes_on:
            blobs = []
            for name, m in masks.items():
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area or area > max_area:
                        continue
                    blobs_seen += 1
                    shape, poly = classify_shape(cnt, shape_strict)
                    if shape is None:
                        continue
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    if any(cv2.pointPolygonTest(op, (float(cx), float(cy)), False) >= 0
                           for op in object_polys):
                        continue   # already part of a recognized object
                    blobs.append({
                        "key": f"blob:{name}", "label": f"{name} {shape}",
                        "bgr": COLORS_BGR[name], "approx": poly,
                        "center": (cx, cy), "area": area, "conf": None,
                    })
            blobs.sort(key=lambda d: d["area"], reverse=True)
            kept = []
            for cand in blobs:
                dup = False
                for k in kept:
                    lim = 0.35 * min(np.sqrt(cand["area"]), np.sqrt(k["area"]))
                    if np.hypot(cand["center"][0] - k["center"][0],
                                cand["center"][1] - k["center"][1]) < lim:
                        dup = True
                        break
                if not dup:
                    kept.append(cand)
            detections.extend(kept)

        # ---- tracking ---------------------------------------------------------
        existing = set(tracks.keys())
        matches, used_d = assign_detections(tracks, detections)
        matched = set()
        for tid, di in matches:
            tracks[tid].update(detections[di])
            matched.add(tid)
        for tid in existing - matched:
            tracks[tid].missed += 1
            if tracks[tid].missed > TRACK_MAX_MISSED:
                del tracks[tid]
        for di, det in enumerate(detections):
            if di in used_d:
                continue
            trk = Track(next_id, det["key"], det["center"])
            trk.update(det)
            tracks[next_id] = trk
            next_id += 1

        # ---- draw the selected tab ----------------------------------------------
        view = state["view"]
        canvas = frame.copy() if view == 0 else black_screen
        active = draw_tracks(canvas, tracks)
        active_set = set(active)

        if active_set != last_announced:
            for item in sorted(active_set - last_announced):
                print(f"{item} detected")
            last_announced = active_set

        if view == 1:
            y = 40
            for item in sorted(active_set):
                cv2.putText(canvas, item, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, NEUTRAL_BGR, 2)
                y += 38

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        if detector is None:
            obj_status = "Objects: OFF - run  pip install ultralytics"
        elif not objects_on:
            obj_status = "Objects: turned off"
        else:
            obj_status = f"Objects: {detector.status}"
        shape_status = (f"Shapes: {len([d for d in detections if d['key'].startswith('blob')])} found "
                        f"({blobs_seen} colored blobs checked)") if shapes_on else "Shapes: turned off"
        status_lines = [shape_status, obj_status, f"FPS {fps:.0f}  |  SPACE = switch view  |  [ ] = resize  |  Q = quit"]

        cv2.imshow(WINDOW, build_ui(canvas, view, layout, status_lines))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (9, 32, ord("v"), ord("V")):      # TAB, SPACE, V = toggle
            state["view"] = 1 - state["view"]
        elif key in (ord("1"), ord("n"), ord("N")):  # 1 or N = normal
            state["view"] = 0
        elif key in (ord("2"), ord("b"), ord("B")):  # 2 or B = black screen
            state["view"] = 1
        elif key == ord("["):                        # [ = shrink window
            win_w = max(600, int(win_w * 0.9))
            cv2.resizeWindow(WINDOW, win_w, int(win_w * ui_h / ui_w))
        elif key == ord("]"):                        # ] = grow window
            win_w = min(2400, int(win_w * 1.1))
            cv2.resizeWindow(WINDOW, win_w, int(win_w * ui_h / ui_w))

        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    if detector is not None:
        detector.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
