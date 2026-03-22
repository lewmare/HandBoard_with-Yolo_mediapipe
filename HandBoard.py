import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from collections import Counter
import mediapipe as mp
import cv2
import numpy as np
import time

# ── Load Model ────────────────────────────────────────────
model_path = hf_hub_download("Bingsu/adetailer", "hand_yolov8n.pt")
yolo       = YOLO(model_path)

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ── Camera ────────────────────────────────────────────────
cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
cap.set(cv2.CAP_PROP_CONTRAST, 128)
cap.set(cv2.CAP_PROP_SHARPNESS, 200)

ret, frame = cap.read()
frame      = cv2.flip(frame, 1)
H, W       = frame.shape[:2]

canvas = np.zeros((H, W, 4), dtype=np.uint8)

# ── Colors ────────────────────────────────────────────────
COLORS = [
    ("Red",    (0,   0,   255, 255)),
    ("Orange", (0,   140, 255, 255)),
    ("Yellow", (0,   220, 255, 255)),
    ("Green",  (0,   220, 0,   255)),
    ("Blue",   (255, 80,  0,   255)),
    ("Purple", (220, 0,   220, 255)),
    ("White",  (255, 255, 255, 255)),
    ("Black",  (0,   0,   0,   255)),
]
color_idx = 0
SIZES     = [2, 4, 8, 16]
size_idx  = 1

# ── Shape objects ─────────────────────────────────────────
shapes = []

def redraw_canvas():
    global canvas
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    for s in shapes:
        col = s["color"]
        t   = s["thickness"]
        if s["type"] == "rect":
            x1, y1, x2, y2 = s["data"]
            cv2.rectangle(canvas, (x1,y1), (x2,y2), col, t)
        elif s["type"] == "circle":
            cx, cy, r = s["data"]
            cv2.circle(canvas, (cx,cy), r, col, t)
        elif s["type"] == "line":
            pts = s["data"]
            for j in range(1, len(pts)):
                p1 = pts[j-1]
                p2 = pts[j]
                d     = dist_pts(p1[0], p1[1], p2[0], p2[1])
                steps = max(1, d // 2)
                for s_i in range(steps + 1):
                    t_i = s_i / steps
                    mx  = int(p1[0] + (p2[0] - p1[0]) * t_i)
                    my  = int(p1[1] + (p2[1] - p1[1]) * t_i)
                    cv2.circle(canvas, (mx, my), t, col, -1)

def point_near_shape(px, py, s, threshold=40):
    if s["type"] == "rect":
        x1, y1, x2, y2 = s["data"]
        inside    = x1 <= px <= x2 and y1 <= py <= y2
        near_edge = (abs(px-x1) < threshold or abs(px-x2) < threshold or
                     abs(py-y1) < threshold or abs(py-y2) < threshold)
        return inside or near_edge
    elif s["type"] == "circle":
        cx, cy, r = s["data"]
        d = dist_pts(px, py, cx, cy)
        return abs(d - r) < threshold or d < threshold
    elif s["type"] == "line":
        for pt in s["data"]:
            if dist_pts(px, py, pt[0], pt[1]) < threshold:
                return True
    return False

def move_shape(idx, dx, dy):
    s = shapes[idx]
    if s["type"] == "rect":
        x1, y1, x2, y2 = s["data"]
        s["data"] = (x1+dx, y1+dy, x2+dx, y2+dy)
    elif s["type"] == "circle":
        cx, cy, r = s["data"]
        s["data"] = (cx+dx, cy+dy, r)
    elif s["type"] == "line":
        s["data"] = [(x+dx, y+dy) for x,y in s["data"]]

# ── State ─────────────────────────────────────────────────
prev_x, prev_y       = None, None
mode                 = "idle"
notification_text    = ""
notification_time    = 0
fist_start_time      = 0
fist_triggered       = False
pinky_start_time     = 0
pinky_triggered      = False
L_HOLD               = 0.6

# Shape state
shape_start       = None
shape_mode        = None
shape_start_time  = 0
shape_last_commit = 0
SHAPE_MIN_HOLD    = 0.3
SHAPE_DEBOUNCE    = 0.5
MIN_SHAPE_SIZE    = 20
DRAW_COOLDOWN     = 0.8
last_shape_time   = 0

# Two finger rect
two_finger_rect_start  = 0
TWO_FINGER_HOLD        = 1.5
two_finger_active      = False
two_finger_committed   = False
two_finger_p1          = None
two_finger_p2          = None
TIPS_CLOSE             = 60

# Drag/move
drag_active   = False
drag_idx      = -1
drag_prev_x   = None
drag_prev_y   = None

# Stabilitas
STABLE_FRAMES    = 10
STABLE_THRESHOLD = 0.015
hand_pos_history = []
hand_is_stable   = False

# Smoothing
SMOOTH_N = 2
smooth_x = []
smooth_y = []

# Frame processing
PROCESS_EVERY  = 1
frame_count    = 0
last_detection = None

# Gesture stability
gesture_history = []
GESTURE_FRAMES  = 3

def stable_gesture(gesture):
    gesture_history.append(gesture)
    if len(gesture_history) > GESTURE_FRAMES:
        gesture_history.pop(0)
    if len(gesture_history) < GESTURE_FRAMES:
        return gesture
    return Counter(gesture_history).most_common(1)[0][0]

# ── Helper functions ──────────────────────────────────────
def smooth_point(x, y):
    smooth_x.append(x)
    smooth_y.append(y)
    if len(smooth_x) > SMOOTH_N:
        smooth_x.pop(0)
        smooth_y.pop(0)
    return int(sum(smooth_x) / len(smooth_x)), \
           int(sum(smooth_y) / len(smooth_y))

def dist(lm, a, b):
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    return (dx**2 + dy**2) ** 0.5

def dist_pts(x1, y1, x2, y2):
    return int(((x2-x1)**2 + (y2-y1)**2) ** 0.5)

def hand_facing_down(lm):
    if len(lm) < 21:
        return False
    return lm[12].y > lm[0].y

def fingers_up(lm):
    if len(lm) < 21:
        return [False] * 5
    facing_down   = hand_facing_down(lm)
    is_right_hand = lm[4].x < lm[20].x
    up = []
    if is_right_hand:
        up.append(lm[4].x < lm[3].x)
    else:
        up.append(lm[4].x > lm[3].x)
    for tip, joint in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if facing_down:
            up.append(lm[tip].y > lm[joint].y)
        else:
            up.append(lm[tip].y < lm[joint].y)
    return up

def update_stability(lm):
    global hand_is_stable
    hand_pos_history.append((lm[0].x, lm[0].y))
    if len(hand_pos_history) > STABLE_FRAMES:
        hand_pos_history.pop(0)
    if len(hand_pos_history) < STABLE_FRAMES:
        hand_is_stable = False
        return
    xs     = [p[0] for p in hand_pos_history]
    ys     = [p[1] for p in hand_pos_history]
    spread = max(max(xs)-min(xs), max(ys)-min(ys))
    hand_is_stable = spread < STABLE_THRESHOLD

def is_fist(lm):
    if len(lm) < 21:
        return False
    facing_down = hand_facing_down(lm)
    for tip, base in zip([8, 12, 16, 20], [5, 9, 13, 17]):
        if facing_down:
            if lm[tip].y > lm[base].y:
                return False
        else:
            if lm[tip].y < lm[base].y:
                return False
    palm_y = lm[9].y
    for tip in [8, 12, 16, 20]:
        if abs(lm[tip].y - palm_y) > 0.15:
            return False
    return True

def is_pinky_only(lm):
    if len(lm) < 21:
        return False
    pinky_len  = dist(lm, 20, 17)
    index_len  = dist(lm, 8,  5)
    middle_len = dist(lm, 12, 9)
    ring_len   = dist(lm, 16, 13)
    if pinky_len < 0.12:
        return False
    if index_len > 0.12 or middle_len > 0.12 or ring_len > 0.12:
        return False
    return True

def is_three_fingers(lm):
    if len(lm) < 21:
        return False
    index_len  = dist(lm, 8,  5)
    middle_len = dist(lm, 12, 9)
    ring_len   = dist(lm, 16, 13)
    pinky_len  = dist(lm, 20, 17)
    if index_len < 0.12 or middle_len < 0.12 or ring_len < 0.12:
        return False
    avg_len = (index_len + middle_len + ring_len) / 3
    if pinky_len > avg_len * 0.6:
        return False
    spread = abs(lm[8].x - lm[16].x)
    if spread > 0.07:
        return False
    return True

def is_four_fingers(up):
    return (not up[0] and up[1] and up[2] and up[3] and up[4])

def is_circle_gesture(lm, up):
    if len(lm) < 21:
        return False
    if not (up[2] and up[3] and up[4]):
        return False
    pinch_dist = dist(lm, 4, 8)
    if pinch_dist > 0.06:
        return False
    is_right_hand = lm[4].x < lm[20].x
    thumb_base_x  = lm[2].x
    index_base_x  = lm[5].x
    if is_right_hand:
        return thumb_base_x < index_base_x
    else:
        return thumb_base_x > index_base_x

def get_shape_anchor(lm):
    ax = int((lm[4].x + lm[8].x) / 2 * W)
    ay = int((lm[4].y + lm[8].y) / 2 * H)
    return ax, ay

def commit_rect(p1, p2):
    global last_shape_time
    if abs(p2[0]-p1[0]) < MIN_SHAPE_SIZE or abs(p2[1]-p1[1]) < MIN_SHAPE_SIZE:
        return False
    col       = COLORS[color_idx][1]
    thickness = SIZES[size_idx]
    shapes.append({
        "type": "rect",
        "color": col,
        "thickness": thickness,
        "data": (p1[0], p1[1], p2[0], p2[1])
    })
    redraw_canvas()
    last_shape_time = time.time()
    return True

def commit_circle(sx, sy, cx, cy):
    global shape_last_commit, last_shape_time
    r = dist_pts(sx, sy, cx, cy)
    if r < MIN_SHAPE_SIZE:
        return False
    col       = COLORS[color_idx][1]
    thickness = SIZES[size_idx]
    shapes.append({
        "type": "circle",
        "color": col,
        "thickness": thickness,
        "data": (sx, sy, r)
    })
    redraw_canvas()
    shape_last_commit = time.time()
    last_shape_time   = time.time()
    return True

def apply_glow(canvas, color_bgra, radius=12):
    alpha = canvas[:, :, 3]
    mask  = (alpha > 0).astype(np.uint8) * 255
    col   = np.array(color_bgra[:3], dtype=np.float32)
    glow  = np.zeros((*mask.shape, 3), dtype=np.float32)
    for r, strength in [(radius, 0.5), (radius // 3, 0.4)]:
        if r < 1:
            r = 1
        b = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r)
        b = b[:, :, np.newaxis] / 255.0
        glow += b * col * strength
    return np.clip(glow, 0, 255).astype(np.uint8)

def set_notif(text):
    global notification_text, notification_time
    notification_text = text
    notification_time = time.time()

def draw_hud(display, now):
    name, bgra = COLORS[color_idx]
    cv2.circle(display, (24, 24), 14, bgra[:3], -1)
    cv2.circle(display, (24, 24), 14, (255, 255, 255), 1)

    if two_finger_active:
        mode_str = "rectangle"
    elif shape_mode == "circle":
        mode_str = "circle"
    elif mode == "draw":
        mode_str = "drawing"
    elif mode == "move":
        mode_str = "moving"
    else:
        mode_str = "idle"

    cv2.putText(display, f"{name}  |  {mode_str}",
                (46, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

    if fist_start_time > 0 and not fist_triggered:
        elapsed  = now - fist_start_time
        progress = min(elapsed / 1.0, 1.0)
        bar_w    = int(200 * progress)
        cv2.rectangle(display, (20, H-28), (220, H-14), (50,50,50), -1)
        cv2.rectangle(display, (20, H-28), (20+bar_w, H-14), (60,60,220), -1)
        cv2.putText(display, "Hold to clear",
                    (20, H-32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (160,160,255), 1)

    if pinky_start_time > 0 and not pinky_triggered:
        elapsed  = now - pinky_start_time
        progress = min(elapsed / L_HOLD, 1.0)
        bar_w    = int(200 * progress)
        next_col = COLORS[(color_idx + 1) % len(COLORS)]
        cv2.rectangle(display, (20, H-52), (220, H-38), (50,50,50), -1)
        cv2.rectangle(display, (20, H-52), (20+bar_w, H-38), next_col[1][:3], -1)
        cv2.putText(display, f"Next: {next_col[0]}",
                    (20, H-56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220,220,160), 1)

    if two_finger_active and not two_finger_committed:
        elapsed  = now - two_finger_rect_start
        progress = min(elapsed / TWO_FINGER_HOLD, 1.0)
        bar_w    = int(200 * progress)
        col      = COLORS[color_idx][1][:3]
        cv2.rectangle(display, (20, H-76), (220, H-62), (50,50,50), -1)
        cv2.rectangle(display, (20, H-76), (20+bar_w, H-62), col, -1)
        cv2.putText(display, "Hold to draw rect...",
                    (20, H-80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220,220,100), 1)

def draw_notif(display, text):
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    x = (W - tw) // 2
    y = H // 2
    cv2.rectangle(display, (x-16, y-th-12), (x+tw+16, y+12), (0,0,0), -1)
    cv2.putText(display, text, (x, y), font, 1.0, (255,255,255), 2)

print("Hand Board — Q: quit | S: save")

cv2.namedWindow("Hand Board", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Board", 1280, 720)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        now   = time.time()
        frame_count += 1

        if frame_count % PROCESS_EVERY == 0:
            small          = cv2.resize(frame, (640, 360))
            rgb            = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image       = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms   = int(time.time() * 1000)
            last_detection = landmarker.detect_for_video(mp_image, timestamp_ms)

        detection  = last_detection
        hand_count = len(detection.hand_landmarks) if detection else 0

        index_tips   = []
        single_hands = []

        if detection:
            for hand_lm in detection.hand_landmarks:
                lm    = hand_lm
                up    = fingers_up(lm)
                tip_x = int(lm[8].x * W)
                tip_y = int(lm[8].y * H)
                single_hands.append((lm, up))
                if up[1] and not up[2] and not up[3] and not up[4]:
                    index_tips.append((tip_x, tip_y, lm, up))

        # ── Mode 2 telunjuk = rectangle ───────────────────
        if len(index_tips) == 2:
            p1        = (index_tips[0][0], index_tips[0][1])
            p2        = (index_tips[1][0], index_tips[1][1])
            tips_dist = dist_pts(p1[0], p1[1], p2[0], p2[1])

            if two_finger_committed:
                if tips_dist > TIPS_CLOSE * 2:
                    two_finger_committed = False
                    two_finger_active    = False
            elif not two_finger_active:
                if tips_dist < TIPS_CLOSE:
                    two_finger_rect_start = now
                    two_finger_active     = True
                    two_finger_p1         = p1
                    two_finger_p2         = p2
                else:
                    cv2.line(frame, p1, p2, (100,100,100), 1)
                    cv2.putText(frame, f"bring closer: {tips_dist}px",
                                (p1[0], p1[1]-15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (150,150,150), 1)
            else:
                two_finger_p1 = p1
                two_finger_p2 = p2
                held          = now - two_finger_rect_start
                col = COLORS[color_idx][1][:3]
                t   = SIZES[size_idx]
                cv2.rectangle(frame, p1, p2, col, t)
                cv2.circle(frame, p1, 8, col, -1)
                cv2.circle(frame, p2, 8, col, -1)
                if held >= TWO_FINGER_HOLD and not two_finger_committed:
                    ok = commit_rect(p1, p2)
                    if ok:
                        set_notif("Rectangle drawn!")
                    two_finger_committed = True

            prev_x, prev_y   = None, None
            fist_start_time  = 0
            fist_triggered   = False
            pinky_start_time = 0
            pinky_triggered  = False
            shape_mode       = None
            shape_start      = None
            mode             = "idle"

        else:
            if two_finger_active and not two_finger_committed:
                if two_finger_p1 and two_finger_p2:
                    held = now - two_finger_rect_start
                    if held >= TWO_FINGER_HOLD:
                        ok = commit_rect(two_finger_p1, two_finger_p2)
                        if ok:
                            set_notif("Rectangle drawn!")
            two_finger_active    = False
            two_finger_committed = False
            two_finger_p1        = None
            two_finger_p2        = None

            if hand_count == 0:
                drag_active      = False
                drag_idx         = -1
                drag_prev_x      = None
                drag_prev_y      = None
                hand_pos_history.clear()
                hand_is_stable   = False
                shape_mode       = None
                shape_start      = None
                fist_start_time  = 0
                fist_triggered   = False
                pinky_start_time = 0
                pinky_triggered  = False
                prev_x, prev_y   = None, None
                smooth_x.clear()
                smooth_y.clear()
                gesture_history.clear()
                mode = "idle"

            for lm, up in single_hands:
                ix = int(lm[8].x * W)
                iy = int(lm[8].y * H)
                ax, ay = get_shape_anchor(lm)
                update_stability(lm)

                # Tentukan gesture saat ini
                if is_fist(lm):
                    current_gesture = "fist"
                elif is_three_fingers(lm):
                    current_gesture = "three"
                elif is_pinky_only(lm):
                    current_gesture = "pinky"
                elif is_four_fingers(up) and len(shapes) > 0:
                    current_gesture = "four"
                elif is_circle_gesture(lm, up):
                    current_gesture = "circle"
                elif up[1] and not up[2] and not up[3] and not up[4]:
                    current_gesture = "draw"
                elif up[1] and up[2]:
                    current_gesture = "lift"
                else:
                    current_gesture = "idle"

                current_gesture = stable_gesture(current_gesture)

                # ── Fist: clear canvas ────────────────────
                if current_gesture == "fist":
                    shape_mode     = None
                    shape_start    = None
                    mode           = "idle"
                    prev_x, prev_y = None, None
                    pinky_start_time = 0
                    pinky_triggered  = False
                    if fist_start_time == 0:
                        fist_start_time = now
                        fist_triggered  = False
                    if now - fist_start_time >= 1.0 and not fist_triggered:
                        canvas = np.zeros((H, W, 4), dtype=np.uint8)
                        shapes.clear()
                        fist_triggered = True
                        set_notif("Canvas cleared")
                    kx = int(lm[9].x * W)
                    ky = int(lm[9].y * H)
                    cv2.circle(frame, (kx, ky), 30, (0,80,255), 2)

                # ── 3 fingers: eraser ─────────────────────
                elif current_gesture == "three":
                    shape_mode     = None
                    shape_start    = None
                    mode           = "idle"
                    prev_x, prev_y = None, None
                    fist_start_time  = 0
                    fist_triggered   = False
                    pinky_start_time = 0
                    pinky_triggered  = False

                    tips_x = [lm[8].x, lm[12].x, lm[16].x]
                    tips_y = [lm[8].y, lm[12].y, lm[16].y]
                    cx_e   = int(sum(tips_x) / 3 * W)
                    cy_e   = int(sum(tips_y) / 3 * H)
                    eraser_r    = 40
                    need_redraw = False
                    to_remove   = []

                    for i, s in enumerate(shapes):
                        if s["type"] == "line":
                            before    = len(s["data"])
                            s["data"] = [
                                pt for pt in s["data"]
                                if dist_pts(cx_e, cy_e, pt[0], pt[1]) > eraser_r
                            ]
                            if len(s["data"]) != before:
                                need_redraw = True
                            if len(s["data"]) == 0:
                                to_remove.append(i)
                        elif s["type"] in ("rect", "circle"):
                            if point_near_shape(cx_e, cy_e, s, threshold=eraser_r):
                                to_remove.append(i)
                                need_redraw = True

                    for i in reversed(to_remove):
                        shapes.pop(i)

                    if need_redraw:
                        redraw_canvas()
                    else:
                        cv2.circle(canvas, (cx_e, cy_e), eraser_r, (0,0,0,0), -1)

                    cv2.circle(frame, (cx_e, cy_e), eraser_r, (255,255,255), 2)
                    cv2.circle(frame, (cx_e, cy_e), 3,        (255,255,255), -1)

                # ── Pinky: next color ──────────────────────
                elif current_gesture == "pinky":
                    shape_mode     = None
                    shape_start    = None
                    mode           = "idle"
                    prev_x, prev_y = None, None
                    fist_start_time = 0
                    fist_triggered  = False
                    if pinky_start_time == 0:
                        pinky_start_time = now
                        pinky_triggered  = False
                    if now - pinky_start_time >= L_HOLD and not pinky_triggered:
                        color_idx       = (color_idx + 1) % len(COLORS)
                        pinky_triggered = True
                        set_notif(COLORS[color_idx][0])
                    px = int(lm[20].x * W)
                    py = int(lm[20].y * H)
                    nc = COLORS[(color_idx + 1) % len(COLORS)][1][:3]
                    cv2.circle(frame, (px, py), 10, nc, -1)
                    cv2.circle(frame, (px, py), 12, (255,255,255), 1)

                # ── 4 jari: move shape ────────────────────
                elif current_gesture == "four":
                    fist_start_time  = 0
                    fist_triggered   = False
                    pinky_start_time = 0
                    pinky_triggered  = False
                    prev_x, prev_y   = None, None
                    smooth_x.clear()
                    smooth_y.clear()
                    mode = "move"

                    mx = int((lm[8].x+lm[12].x+lm[16].x+lm[20].x)/4 * W)
                    my = int((lm[8].y+lm[12].y+lm[16].y+lm[20].y)/4 * H)

                    if not drag_active:
                        for i, s in enumerate(shapes):
                            if s["type"] in ("rect", "circle") and \
                               point_near_shape(mx, my, s):
                                drag_active = True
                                drag_idx    = i
                                drag_prev_x = mx
                                drag_prev_y = my
                                set_notif("Shape selected!")
                                break

                    if drag_active and drag_idx >= 0:
                        if drag_prev_x is not None:
                            dx = mx - drag_prev_x
                            dy = my - drag_prev_y
                            move_shape(drag_idx, dx, dy)
                            redraw_canvas()
                        drag_prev_x = mx
                        drag_prev_y = my
                        s = shapes[drag_idx]
                        if s["type"] == "rect":
                            x1,y1,x2,y2 = s["data"]
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                        elif s["type"] == "circle":
                            cx,cy,r = s["data"]
                            cv2.circle(frame, (cx,cy), r, (0,255,255), 2)

                    cv2.circle(frame, (mx, my), 12, (0,255,255), 2)

                # ── Circle gesture ────────────────────────
                elif current_gesture == "circle":
                    fist_start_time  = 0
                    fist_triggered   = False
                    pinky_start_time = 0
                    pinky_triggered  = False
                    prev_x, prev_y   = None, None
                    mode = "idle"
                    if shape_mode != "circle":
                        if shape_mode is not None and shape_start is not None:
                            if now - shape_start_time >= SHAPE_MIN_HOLD:
                                if now - shape_last_commit >= SHAPE_DEBOUNCE:
                                    commit_circle(shape_start[0], shape_start[1],
                                                  ax, ay)
                        shape_mode       = "circle"
                        shape_start      = (ax, ay)
                        shape_start_time = now
                    if now - shape_start_time >= SHAPE_MIN_HOLD and shape_start:
                        col = COLORS[color_idx][1][:3]
                        t   = SIZES[size_idx]
                        r   = dist_pts(shape_start[0], shape_start[1], ax, ay)
                        cv2.circle(frame, shape_start, r, col, t)
                        cv2.circle(frame, shape_start, 6, (255,255,255), 2)
                    cv2.circle(frame, (ax, ay), 8, COLORS[color_idx][1][:3], -1)

                else:
                    # Reset drag saat gesture dilepas
                    if drag_active:
                        drag_active = False
                        drag_idx    = -1
                        drag_prev_x = None
                        drag_prev_y = None

                    if shape_mode is not None and shape_start is not None:
                        if now - shape_start_time >= SHAPE_MIN_HOLD:
                            if now - shape_last_commit >= SHAPE_DEBOUNCE:
                                ok = commit_circle(shape_start[0], shape_start[1],
                                                   ax, ay)
                                if ok:
                                    set_notif("Circle drawn!")
                    shape_mode  = None
                    shape_start = None
                    fist_start_time  = 0
                    fist_triggered   = False
                    pinky_start_time = 0
                    pinky_triggered  = False

                    # ── Draw ──────────────────────────────
                    if current_gesture == "draw" and \
                       now - last_shape_time > DRAW_COOLDOWN:
                        mode      = "draw"
                        col_bgra  = COLORS[color_idx][1]
                        thickness = SIZES[size_idx]
                        sx, sy    = smooth_point(ix, iy)

                        if prev_x is not None:
                            jump_dist = dist_pts(prev_x, prev_y, sx, sy)
                            if jump_dist > 150:
                                prev_x, prev_y = sx, sy
                                smooth_x.clear()
                                smooth_y.clear()
                                shapes.append({
                                    "type": "line",
                                    "color": col_bgra,
                                    "thickness": thickness,
                                    "data": [(sx, sy)]
                                })
                            else:
                                d     = dist_pts(prev_x, prev_y, sx, sy)
                                steps = max(1, d // 2)
                                for s_i in range(steps + 1):
                                    t_i = s_i / steps
                                    mx  = int(prev_x + (sx - prev_x) * t_i)
                                    my  = int(prev_y + (sy - prev_y) * t_i)
                                    cv2.circle(canvas, (mx, my),
                                               thickness, col_bgra, -1)
                                if not shapes or shapes[-1]["type"] != "line" or \
                                   shapes[-1]["color"] != col_bgra:
                                    shapes.append({
                                        "type": "line",
                                        "color": col_bgra,
                                        "thickness": thickness,
                                        "data": []
                                    })
                                shapes[-1]["data"].append((sx, sy))
                        else:
                            shapes.append({
                                "type": "line",
                                "color": col_bgra,
                                "thickness": thickness,
                                "data": [(sx, sy)]
                            })

                        prev_x, prev_y = sx, sy
                        cv2.circle(frame, (sx, sy), thickness+4, col_bgra[:3], -1)
                        cv2.circle(frame, (sx, sy), thickness+6, (255,255,255), 1)

                    elif current_gesture == "lift":
                        mode           = "idle"
                        prev_x, prev_y = None, None
                        smooth_x.clear()
                        smooth_y.clear()

                    else:
                        prev_x, prev_y = None, None

        # ── Blend canvas + glow ───────────────────────────
        canvas_bgr = canvas[:, :, :3]
        alpha      = canvas[:, :, 3:4].astype(float) / 255.0

        display = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)

        has_drawing = np.any(canvas[:, :, 3] > 0)
        if has_drawing:
            if mode == "draw":
                glow    = apply_glow(canvas, COLORS[color_idx][1])
                display = np.clip(
                    display.astype(float) + glow.astype(float) * 0.8,
                    0, 255
                ).astype(np.uint8)
            mask          = alpha[:, :, 0] > 0
            display[mask] = canvas_bgr[mask]

        draw_hud(display, now)

        if now - notification_time < 1.5:
            draw_notif(display, notification_text)

        cv2.imshow("Hand Board", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"glass_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            set_notif(f"Saved: {fname}")

cap.release()
cv2.destroyAllWindows()