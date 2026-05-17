"""
Driver Drowsiness Detector
===========================
Real-time drowsiness detection using:
- OpenCV for webcam feed
- Dlib for facial landmark detection
- Custom CNN model (trained via train.py)
- Eye Aspect Ratio (EAR) algorithm

Run: python detect.py
"""

import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
from tensorflow.keras.models import load_model
import pygame
import time
import os

# ─── Constants ────────────────────────────────────────────────────────────────
EAR_THRESHOLD    = 0.25   # Below this → eye is "closed"
EAR_CONSEC_FRAMES = 20    # Consecutive frames before alert triggers
ALARM_SOUND      = "alarm.wav"
MODEL_PATH       = "drowsiness_model.h5"
PREDICTOR_PATH   = "shape_predictor_68_face_landmarks.dat"

# ─── Init ──────────────────────────────────────────────────────────────────────
pygame.mixer.init()

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Load CNN model if available (used as secondary classifier)
cnn_model = None
if os.path.exists(MODEL_PATH):
    cnn_model = load_model(MODEL_PATH)
    print("[INFO] CNN model loaded.")
else:
    print("[WARNING] CNN model not found. Using EAR-only detection.")

# ─── Helper Functions ──────────────────────────────────────────────────────────
def eye_aspect_ratio(eye):
    """Compute Eye Aspect Ratio (EAR) for one eye."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def preprocess_eye(frame, eye_points):
    """Crop and preprocess eye region for CNN inference."""
    x_coords = [p[0] for p in eye_points]
    y_coords = [p[1] for p in eye_points]
    x1, x2 = max(min(x_coords) - 5, 0), min(max(x_coords) + 5, frame.shape[1])
    y1, y2 = max(min(y_coords) - 5, 0), min(max(y_coords) + 5, frame.shape[0])
    eye_roi = frame[y1:y2, x1:x2]
    if eye_roi.size == 0:
        return None
    eye_roi = cv2.resize(eye_roi, (24, 24))
    eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    eye_roi = eye_roi.astype("float32") / 255.0
    eye_roi = np.expand_dims(eye_roi, axis=(0, -1))
    return eye_roi


def play_alarm():
    """Play alarm sound (non-blocking)."""
    if os.path.exists(ALARM_SOUND):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(ALARM_SOUND)
            pygame.mixer.music.play(-1)
    else:
        print("\a")  # System beep fallback


def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


def draw_ui(frame, ear, frame_count, drowsy, fps):
    """Render HUD overlay on frame."""
    h, w = frame.shape[:2]

    # Status badge
    if drowsy:
        color   = (0, 0, 255)
        label   = "DROWSY - WAKE UP!"
        bg_color = (0, 0, 180)
    else:
        color   = (0, 220, 100)
        label   = "ALERT"
        bg_color = (0, 120, 50)

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.putText(frame, "Driver Drowsiness Detector", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    # Status badge (bottom-left)
    cv2.rectangle(frame, (0, h - 55), (w, h), bg_color, -1)
    cv2.putText(frame, label, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # EAR + FPS (right side)
    cv2.putText(frame, f"EAR: {ear:.2f}", (w - 160, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 160, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # EAR bar
    bar_x, bar_y, bar_h = w - 30, 60, h - 120
    filled = int(bar_h * min(ear / 0.45, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_h), (50, 50, 50), -1)
    bar_color = (0, 220, 100) if ear > EAR_THRESHOLD else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y + bar_h - filled),
                  (bar_x + 20, bar_y + bar_h), bar_color, -1)

    return frame


# ─── Main Loop ────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    COUNTER  = 0
    ALARM_ON = False
    prev_time = time.time()

    print("[INFO] Starting detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = np.uint8(gray)
        rects = detector(gray, 0)

        ear   = 0.0
        drowsy = False

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye  = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR  = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 200, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 200, 255), 1)

            # ── CNN secondary check ──────────────────────────────────────────
            cnn_closed = False
            if cnn_model is not None:
                left_roi  = preprocess_eye(frame, leftEye)
                right_roi = preprocess_eye(frame, rightEye)
                if left_roi is not None and right_roi is not None:
                    l_pred = cnn_model.predict(left_roi,  verbose=0)[0][0]
                    r_pred = cnn_model.predict(right_roi, verbose=0)[0][0]
                    cnn_closed = (l_pred > 0.5 and r_pred > 0.5)

            # ── Decision logic ───────────────────────────────────────────────
            if ear < EAR_THRESHOLD or cnn_closed:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    drowsy   = True
                    ALARM_ON = True
                    play_alarm()
            else:
                COUNTER  = 0
                ALARM_ON = False
                stop_alarm()

        frame = draw_ui(frame, ear, COUNTER, drowsy, fps)
        cv2.imshow("Driver Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")


if __name__ == "__main__":
    main()
