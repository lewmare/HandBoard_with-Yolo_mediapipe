import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
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
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# ── Camera ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

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
shape_start          = None
shape_mode           = None
prev_shape_mode      = None
shape_start_time     = 0
SHAPE_MIN_HOLD       = 0.3
shape_last_commit    = 0       # waktu terakhir shape di-commit
SHAPE_DEBOUNCE       = 0.5     # jeda minimum antar commit

# Smoothing
SMOOTH_N = 2
smooth_x = []
smooth_y = []

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

def is_rect_gesture(lm, up):
    """
    Kotak aktif ketika:
    - Hanya telunjuk + ibu jari terangkat
    - 3 jari lain menekuk
    - X ujung ibu jari dan telunjuk hampir sama (toleransi 0.08)
    - Y ujung ibu jari dan telunjuk JAUH (belum merapat = sedang sizing)
    """
    if len(lm) < 21:
        return False

    if not (up[0] and up[1]):
        return False

    # 3 jari lain harus menekuk
    middle_len = dist(lm, 12, 9)
    ring_len   = dist(lm, 16, 13)
    pinky_len  = dist(lm, 20, 17)
    index_len  = dist(lm, 8,  5)
    threshold  = index_len * 0.6
    if middle_len > threshold or ring_len > threshold or pinky_len > threshold:
        return False

    # X ujung ibu jari dan telunjuk harus sejajar
    thumb_x = lm[4].x
    index_x = lm[8].x
    if abs(thumb_x - index_x) > 0.08:
        return False

    # Y harus JAUH — belum merapat (masih sizing)
    thumb_y = lm[4].y
    index_y = lm[8].y
    if abs(thumb_y - index_y) < 0.06:
        return False

    return True

def is_rect_commit(lm, up):
    """
    Commit kotak ketika:
    - Pose sama dengan rect gesture
    - Tapi Y ujung ibu jari dan telunjuk MERAPAT
    """
    if len(lm) < 21:
        return False

    if not (up[0] and up[1]):
        return False

    middle_len = dist(lm, 12, 9)
    ring_len   = dist(lm, 16, 13)
    pinky_len  = dist(lm, 20, 17)
    index_len  = dist(lm, 8,  5)
    threshold  = index_len * 0.6
    if middle_len > threshold or ring_len > threshold or pinky_len > threshold:
        return False

    # X sejajar
    thumb_x = lm[4].x
    index_x = lm[8].x
    if abs(thumb_x - index_x) > 0.08:
        return False

    # Y MERAPAT = selesai
    thumb_y = lm[4].y
    index_y = lm[8].y
    return abs(thumb_y - index_y) < 0.06

def is_circle_gesture(lm, up):
    """
    Lingkaran: pose OK — ibu jari + telunjuk PINCH (dekat),
    jari tengah/manis/kelingking terangkat.
    """
    if len(lm) < 21:
        return False

    # Jari tengah, manis, kelingking harus terangkat
    if not (up[2] and up[3] and up[4]):
        return False

    # Ibu jari & telunjuk harus berdekatan (pinch)
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
    """Titik anchor = tengah antara ibu jari dan telunjuk"""
    ax = int((lm[4].x + lm[8].x) / 2 * W)
    ay = int((lm[4].y + lm[8].y) / 2 * H)
    return ax, ay

def draw_shape_preview(frame_copy, sx, sy, cx, cy, shape):
    """Gambar preview shape di frame (bukan canvas)"""
    col = COLORS[color_idx][1][:3]
    thickness = SIZES[size_idx]
    if shape == "rect":
        cv2.rectangle(frame_copy, (sx, sy), (cx, cy), col, thickness)
    elif shape == "circle":
        r = dist_pts(sx, sy, cx, cy)
        cv2.circle(frame_copy, (sx, sy), r, col, thickness)

MIN_SHAPE_SIZE = 20  # pixel minimum agar shape tidak terbentuk tidak sengaja

def commit_shape(sx, sy, cx, cy, shape):
    """Tulis shape ke canvas — hanya jika ukurannya cukup besar"""
    col       = COLORS[color_idx][1]
    thickness = SIZES[size_idx]
    if shape == "rect":
        # Cek lebar dan tinggi minimum
        if abs(cx - sx) < MIN_SHAPE_SIZE or abs(cy - sy) < MIN_SHAPE_SIZE:
            return
        cv2.rectangle(canvas, (sx, sy), (cx, cy), col, thickness)
    elif shape == "circle":
        r = dist_pts(sx, sy, cx, cy)
        if r < MIN_SHAPE_SIZE:
            return
        cv2.circle(canvas, (sx, sy), r, col, thickness)

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

    if shape_mode == "rect":
        mode_str = "rectangle"
    elif shape_mode == "circle":
        mode_str = "circle"
    elif mode == "draw":
        mode_str = "drawing"
    else:
        mode_str = "idle"

    cv2.putText(display, f"{name}  |  {mode_str}",
                (46, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

    # Fist progress bar
    if fist_start_time > 0 and not fist_triggered:
        elapsed  = now - fist_start_time
        progress = min(elapsed / 1.0, 1.0)
        bar_w    = int(200 * progress)
        cv2.rectangle(display, (20, H-28), (220, H-14), (50, 50, 50), -1)
        cv2.rectangle(display, (20, H-28), (20+bar_w, H-14), (60, 60, 220), -1)
        cv2.putText(display, "Hold to clear",
                    (20, H-32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (160, 160, 255), 1)

    # Pinky progress bar
    if pinky_start_time > 0 and not pinky_triggered:
        elapsed  = now - pinky_start_time
        progress = min(elapsed / L_HOLD, 1.0)
        bar_w    = int(200 * progress)
        next_col = COLORS[(color_idx + 1) % len(COLORS)]
        cv2.rectangle(display, (20, H-52), (220, H-38), (50, 50, 50), -1)
        cv2.rectangle(display, (20, H-52), (20+bar_w, H-38), next_col[1][:3], -1)
        cv2.putText(display, f"Next: {next_col[0]}",
                    (20, H-56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 220, 160), 1)

def draw_notif(display, text):
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    x = (W - tw) // 2
    y = H // 2
    cv2.rectangle(display, (x-16, y-th-12), (x+tw+16, y+12), (0, 0, 0), -1)
    cv2.putText(display, text, (x, y), font, 1.0, (255, 255, 255), 2)

print("Hand Board — Q: quit | S: save")
print("☝  1 jari          = draw")
print("☝👍 telunjuk+jempol = rectangle")
print("👌 pose OK          = circle")
print("kelingking          = next color")
print("3 jari rapat        = eraser")
print("kepalan tahan 1s    = clear")

# Setup window sekali saja sebelum loop
cv2.namedWindow("Hand Board", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Board", W, H)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        now   = time.time()

        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        detection    = landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_detected = len(detection.hand_landmarks) > 0

        if not hand_detected:
            # Gesture dilepas — commit shape jika ada
            if shape_mode is not None and shape_start is not None:
                commit_shape(shape_start[0], shape_start[1],
                             shape_cur[0],   shape_cur[1],
                             shape_mode)
                set_notif(f"{shape_mode.capitalize()} drawn!")
            shape_mode       = None
            shape_start      = None
            fist_start_time  = 0
            fist_triggered   = False
            pinky_start_time = 0
            pinky_triggered  = False
            prev_x, prev_y   = None, None
            mode = "idle"

        # Titik akhir shape sementara (update tiap frame)
        shape_cur = (0, 0)

        for hand_lm in detection.hand_landmarks:
            lm = hand_lm
            up = fingers_up(lm)
            ix = int(lm[8].x * W)
            iy = int(lm[8].y * H)
            ax, ay = get_shape_anchor(lm)
            shape_cur = (ax, ay)

            # ── Fist: clear canvas ────────────────────────
            if is_fist(lm):
                # Commit shape dulu jika ada
                if shape_mode is not None and shape_start is not None:
                    commit_shape(shape_start[0], shape_start[1],
                                 ax, ay, shape_mode)
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
                    canvas         = np.zeros((H, W, 4), dtype=np.uint8)
                    fist_triggered = True
                    set_notif("Canvas cleared")

                kx = int(lm[9].x * W)
                ky = int(lm[9].y * H)
                cv2.circle(frame, (kx, ky), 30, (0, 80, 255), 2)

            # ── 3 fingers: eraser ─────────────────────────
            elif is_three_fingers(lm):
                if shape_mode is not None and shape_start is not None:
                    commit_shape(shape_start[0], shape_start[1],
                                 ax, ay, shape_mode)
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
                cv2.circle(canvas, (cx_e, cy_e), 40, (0, 0, 0, 0), -1)
                cv2.circle(frame,  (cx_e, cy_e), 40, (255, 255, 255), 2)
                cv2.circle(frame,  (cx_e, cy_e), 3,  (255, 255, 255), -1)

            # ── Pinky only: next color ─────────────────────
            elif is_pinky_only(lm):
                if shape_mode is not None and shape_start is not None:
                    commit_shape(shape_start[0], shape_start[1],
                                 ax, ay, shape_mode)
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
                cv2.circle(frame, (px, py), 12, (255, 255, 255), 1)

            # ── Rectangle gesture ─────────────────────────
            elif is_rect_commit(lm, up) and shape_mode == "rect":
                # Y merapat = commit kotak
                if shape_start is not None:
                    if now - shape_last_commit >= SHAPE_DEBOUNCE:
                        commit_shape(shape_start[0], shape_start[1],
                                     ax, ay, "rect")
                        set_notif("Rectangle drawn!")
                        shape_last_commit = now
                shape_mode       = None
                shape_start      = None
                fist_start_time  = 0
                fist_triggered   = False
                pinky_start_time = 0
                pinky_triggered  = False
                prev_x, prev_y   = None, None
                mode = "idle"

            elif is_rect_gesture(lm, up):
                fist_start_time  = 0
                fist_triggered   = False
                pinky_start_time = 0
                pinky_triggered  = False
                prev_x, prev_y   = None, None
                mode = "idle"

                if shape_mode != "rect":
                    if shape_mode is not None and shape_start is not None:
                        commit_shape(shape_start[0], shape_start[1],
                                     ax, ay, shape_mode)
                    shape_mode       = "rect"
                    shape_start      = (ax, ay)
                    shape_start_time = now

                # Preview real-time
                col = COLORS[color_idx][1][:3]
                t   = SIZES[size_idx]
                if shape_start is not None:
                    cv2.rectangle(frame, shape_start, (ax, ay), col, t)
                cv2.circle(frame, (ax, ay), 8, col, -1)
                if shape_start:
                    cv2.circle(frame, shape_start, 6, (255,255,255), 2)

            # ── Circle gesture ────────────────────────────
            elif is_circle_gesture(lm, up):
                fist_start_time  = 0
                fist_triggered   = False
                pinky_start_time = 0
                pinky_triggered  = False
                prev_x, prev_y   = None, None
                mode = "idle"

                if shape_mode != "circle":
                    if shape_mode is not None and shape_start is not None:
                        commit_shape(shape_start[0], shape_start[1],
                                     ax, ay, shape_mode)
                    shape_mode       = "circle"
                    shape_start      = (ax, ay)
                    shape_start_time = now

                # Preview hanya setelah tahan minimum
                if now - shape_start_time >= SHAPE_MIN_HOLD:
                    col = COLORS[color_idx][1][:3]
                    t   = SIZES[size_idx]
                    r   = dist_pts(shape_start[0], shape_start[1], ax, ay)
                    cv2.circle(frame, shape_start, r, col, t)
                    cv2.circle(frame, (ax, ay), 8, col, -1)
                    cv2.circle(frame, shape_start, 6, (255,255,255), 2)

                # Visual titik anchor (selalu tampil)
                cv2.circle(frame, (ax, ay), 8, COLORS[color_idx][1][:3], -1)
                cv2.circle(frame, shape_start, 6, (255,255,255), 2)

            else:
                if shape_mode is not None and shape_start is not None:
                    if now - shape_start_time >= SHAPE_MIN_HOLD:
                        if now - shape_last_commit >= SHAPE_DEBOUNCE:
                            commit_shape(shape_start[0], shape_start[1],
                                         ax, ay, shape_mode)
                            set_notif(f"{shape_mode.capitalize()} drawn!")
                            shape_last_commit = now

                shape_mode  = None
                shape_start = None

                fist_start_time  = 0
                fist_triggered   = False
                pinky_start_time = 0
                pinky_triggered  = False

                # ── 1 finger: draw ────────────────────────
                if up[1] and not up[2] and not up[3] and not up[4]:
                    mode      = "draw"
                    col_bgra  = COLORS[color_idx][1]
                    thickness = SIZES[size_idx]
                    sx, sy    = smooth_point(ix, iy)

                    if prev_x is not None:
                        cv2.line(canvas, (prev_x, prev_y), (sx, sy),
                                 col_bgra, thickness * 2)
                        cv2.line(canvas, (prev_x, prev_y), (sx, sy),
                                 col_bgra, max(1, thickness - 1))

                    prev_x, prev_y = sx, sy
                    cv2.circle(frame, (sx, sy), thickness+4, col_bgra[:3], -1)
                    cv2.circle(frame, (sx, sy), thickness+6, (255, 255, 255), 1)

                # ── 2+ fingers: lift pen ──────────────────
                elif up[1] and up[2]:
                    mode           = "idle"
                    prev_x, prev_y = None, None
                    smooth_x.clear()
                    smooth_y.clear()

                else:
                    prev_x, prev_y = None, None

        # ── Blend canvas + glow onto camera feed ──────────
        canvas_bgr = canvas[:, :, :3]
        alpha      = canvas[:, :, 3:4].astype(float) / 255.0
        display    = frame.copy()

        has_drawing = np.any(canvas[:, :, 3] > 0)
        if has_drawing:
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