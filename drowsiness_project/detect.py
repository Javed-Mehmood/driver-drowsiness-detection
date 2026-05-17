"""
Driver Drowsiness Detector
Fixed: Strict eye detection to avoid false positives
"""

import cv2
import numpy as np
import pygame
import time
import os

ALARM_SOUND   = "alarm.wav"
CONSEC_FRAMES = 15

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

pygame.mixer.init()

def play_alarm():
    if os.path.exists(ALARM_SOUND) and not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(ALARM_SOUND)
        pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

COUNTER   = 0
prev_time = time.time()
print("[INFO] Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w   = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = int(np.mean(gray))

    # Adaptive CLAHE
    clip      = 2.0 if bright > 150 else (3.0 if bright > 100 else 5.0)
    clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    enhanced  = clahe_obj.apply(gray)

    # FPS
    curr_time = time.time()
    fps       = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    faces      = face_cascade.detectMultiScale(enhanced, 1.1, 5, minSize=(80,80))
    face_found = len(faces) > 0
    eyes_found = 0
    drowsy     = False

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (255,200,0), 2)

        # ── Strict eye region ─────────────────────────────
        # Only top 45% of face — avoids nose/cheek false detections
        eye_y1 = fy + int(fh * 0.08)  # skip forehead top
        eye_y2 = fy + int(fh * 0.50)  # stop before nose
        eye_x1 = fx
        eye_x2 = fx + fw

        roi_gray  = enhanced[eye_y1:eye_y2, eye_x1:eye_x2]
        roi_color = frame[eye_y1:eye_y2, eye_x1:eye_x2]

        roi_h, roi_w = roi_gray.shape[:2]

        # Min eye size = 15% of face width
        min_eye = max(12, int(fw * 0.12))
        # Max eye size = 40% of face width (filter out large false detections)
        max_eye = int(fw * 0.40)

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=6,      # HIGH minNeighbors = strict, fewer false positives
            minSize=(min_eye, min_eye),
            maxSize=(max_eye, max_eye)
        )

        # Filter: keep only top 2 valid eyes
        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Eye must be in upper portion of roi
            if ey < roi_h * 0.8:
                valid_eyes.append((ex, ey, ew, eh))

        valid_eyes = valid_eyes[:2]
        eyes_found = len(valid_eyes)

        for (ex, ey, ew, eh) in valid_eyes:
            cx = ex + ew // 2
            cy = ey + eh // 2
            cv2.circle(roi_color, (cx, cy), ew//2, (0,220,255), 2)
            cv2.putText(roi_color, "OPEN",
                        (ex, max(0,ey-4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0,220,255), 1)

    # ── Drowsiness Logic ─────────────────────────────────
    if face_found:
        if eyes_found == 0:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES:
                drowsy = True
                play_alarm()
        else:
            COUNTER = max(0, COUNTER - 2)
            if COUNTER == 0:
                stop_alarm()
    else:
        COUNTER = max(0, COUNTER - 1)
        if COUNTER == 0:
            stop_alarm()

    # ── HUD ──────────────────────────────────────────────
    bg_c = (0,0,180) if drowsy else (0,130,50)
    txt  = "DROWSY - WAKE UP!" if drowsy else "ALERT - Eyes Open"

    cv2.rectangle(frame, (0,0), (w,44), (20,20,20), -1)
    cv2.putText(frame, "Driver Drowsiness Detector",
                (8,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (220,220,220), 2)
    cv2.rectangle(frame, (0,h-44), (w,h), bg_c, -1)
    cv2.putText(frame, txt, (8,h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame,
                f"Eyes:{eyes_found} | Counter:{COUNTER}/{CONSEC_FRAMES} | Bright:{bright} | FPS:{fps:.0f}",
                (8,64), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180,255,180), 1)

    cv2.imshow("Driver Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stop_alarm()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
