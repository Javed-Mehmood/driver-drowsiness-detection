"""
Driver Drowsiness Detector - Final GUI
Fixed: Camera size + Same detection as detect.py
Run: python app.py
"""

import cv2
import numpy as np
import pygame
import time
import os
import threading
import tkinter as tk
from PIL import Image, ImageTk

ALARM_SOUND   = "alarm.wav"

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


class DrowsinessApp:
    def __init__(self, root):
        self.root         = root
        self.root.title("Driver Drowsiness Detector")
        self.root.configure(bg="#0e1117")
        self.root.geometry("1280x720")
        self.root.state("zoomed")

        self.running      = False
        self.counter      = 0
        self.drowsy_count = 0
        self.fps          = 0
        self.cap          = None
        self.log_entries  = []

        # Fixed camera display size — never changes
        self.CAM_W = 860
        self.CAM_H = 520

        self._build_ui()
        self.root.after(500, self.start_detection)

    def _build_ui(self):
        # Title
        title = tk.Frame(self.root, bg="#161b22", pady=8)
        title.pack(fill="x")
        tk.Label(title, text="🚗  Driver Drowsiness Detector",
                 font=("Arial", 18, "bold"), fg="#00d4aa",
                 bg="#161b22").pack(side="left", padx=15)
        tk.Label(title,
                 text="Javed Mehmood (2023-BS-AI-029)  |  Muneeb Sajid (2023-BS-AI-028)  |  BS AI 6th Sem 2026",
                 font=("Arial", 9), fg="#888",
                 bg="#161b22").pack(side="right", padx=15)

        # Main
        main = tk.Frame(self.root, bg="#0e1117")
        main.pack(fill="both", expand=True, padx=8, pady=5)

        # ── LEFT: Camera fixed size ───────────────────────
        left = tk.Frame(main, bg="#0e1117")
        left.pack(side="left", fill="y", padx=(0,8))

        # Fixed size camera label — never resizes
        self.camera_label = tk.Label(
            left, bg="#1e2130",
            width=self.CAM_W, height=self.CAM_H)
        self.camera_label.pack_propagate(False)
        self.camera_label.pack()

        self.status_label = tk.Label(
            left, text="⏸  STARTING...",
            font=("Arial", 20, "bold"),
            fg="white", bg="#333333",
            pady=10, width=60)
        self.status_label.pack(fill="x", pady=(4,0))

        # Info bar
        info_bar = tk.Frame(left, bg="#0e1117")
        info_bar.pack(fill="x", pady=(4,0))
        self.info_lbl = tk.Label(
            info_bar,
            text="Eyes:0 | Counter:0 | FPS:0",
            font=("Arial", 10), fg="#aaa", bg="#0e1117")
        self.info_lbl.pack(side="left")

        # ── RIGHT: Controls fixed width ───────────────────
        right = tk.Frame(main, bg="#0e1117", width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        btn_s = {"relief": "flat", "cursor": "hand2",
                 "pady": 10, "font": ("Arial", 12, "bold")}
        tk.Button(right, text="▶  Start",
                  command=self.start_detection,
                  bg="#00d4aa", fg="white",
                  **btn_s).pack(fill="x", pady=3, padx=6)
        tk.Button(right, text="⏹  Stop",
                  command=self.stop_detection,
                  bg="#ff4b2b", fg="white",
                  **btn_s).pack(fill="x", pady=3, padx=6)
        tk.Button(right, text="🔄  Reset Counter",
                  command=self.reset_counter,
                  bg="#444", fg="white",
                  font=("Arial", 10), relief="flat",
                  cursor="hand2", pady=7
                  ).pack(fill="x", pady=3, padx=6)

        tk.Label(right, text="⚙  Alert Sensitivity",
                 font=("Arial", 11, "bold"),
                 fg="#00d4aa", bg="#0e1117").pack(pady=(12,2))
        self.sens_var = tk.IntVar(value=15)
        tk.Scale(right, from_=5, to=30, orient="horizontal",
                 variable=self.sens_var, bg="#1e2130", fg="white",
                 highlightthickness=0, troughcolor="#333",
                 activebackground="#00d4aa",
                 length=255).pack(padx=6)
        tk.Label(right, text="Low=Sensitive  |  High=Stable",
                 font=("Arial", 8), fg="#666",
                 bg="#0e1117").pack()

        # Metrics
        tk.Label(right, text="📊  Live Metrics",
                 font=("Arial", 11, "bold"),
                 fg="#00d4aa", bg="#0e1117").pack(pady=(14,5))

        mf = tk.Frame(right, bg="#0e1117")
        mf.pack(fill="x", padx=6)
        r1 = tk.Frame(mf, bg="#0e1117"); r1.pack(fill="x", pady=2)
        r2 = tk.Frame(mf, bg="#0e1117"); r2.pack(fill="x", pady=2)

        self.counter_val = self._metric(r1, "Counter",       "0", "#ff6b6b", "left")
        self.eyes_val    = self._metric(r1, "Eyes",          "0", "#00d4aa", "right")
        self.fps_val     = self._metric(r2, "FPS",           "0", "#ffd700", "left")
        self.drowsy_val  = self._metric(r2, "Drowsy Events", "0", "#ff4b2b", "right")

        tk.Label(right, text="🕐  Event Log",
                 font=("Arial", 11, "bold"),
                 fg="#00d4aa", bg="#0e1117").pack(pady=(14,3))
        self.log_box = tk.Text(right, bg="#1e2130", fg="#aaa",
                               font=("Courier", 9), relief="flat",
                               state="disabled", height=10)
        self.log_box.pack(fill="x", padx=6, pady=3)

        tk.Label(right,
                 text="ANN & Deep Learning\nDr. Muhammad Gufran Khan",
                 font=("Arial", 8), fg="#444",
                 bg="#0e1117", justify="center").pack(pady=(8,0))

    def _metric(self, parent, label, val, color, side):
        f = tk.Frame(parent, bg="#1e2130", pady=7, padx=6)
        f.pack(side=side, fill="both", expand=True, padx=2)
        v = tk.Label(f, text=val, font=("Arial", 22, "bold"),
                     fg=color, bg="#1e2130")
        v.pack()
        tk.Label(f, text=label, font=("Arial", 8),
                 fg="#888", bg="#1e2130").pack()
        return v

    def start_detection(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            threading.Thread(target=self._detect_loop, daemon=True).start()

    def stop_detection(self):
        self.running = False
        stop_alarm()
        self.status_label.config(text="⏸  PAUSED", bg="#333333")
        self.camera_label.config(image="", bg="#1e2130")

    def reset_counter(self):
        self.counter = 0

    def _detect_loop(self):
        prev_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            h, w   = frame.shape[:2]
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright = int(np.mean(gray))

            # Adaptive CLAHE
            clip      = 2.0 if bright > 150 else (3.0 if bright > 100 else 5.0)
            clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
            enhanced  = clahe_obj.apply(gray)

            curr_time = time.time()
            self.fps  = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            thresh     = self.sens_var.get()
            faces      = face_cascade.detectMultiScale(enhanced, 1.1, 5, minSize=(80,80))
            face_found = len(faces) > 0
            eyes_found = 0
            drowsy     = False

            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (255,200,0), 2)

                # Strict eye region: 10%-45% of face height
                eye_y1    = fy + int(fh * 0.08)
                eye_y2    = fy + int(fh * 0.52)
                roi_gray  = enhanced[eye_y1:eye_y2, fx:fx+fw]
                roi_color = frame[eye_y1:eye_y2, fx:fx+fw]
                roi_h     = roi_gray.shape[0]

                min_eye = max(10, int(fw * 0.10))
                max_eye = int(fw * 0.40)

                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.05,
                    minNeighbors=5,
                    minSize=(min_eye, min_eye),
                    maxSize=(max_eye, max_eye)
                )

                valid = [(ex,ey,ew,eh) for (ex,ey,ew,eh) in eyes
                         if ey < roi_h * 0.8][:2]
                eyes_found = len(valid)

                for (ex,ey,ew,eh) in valid:
                    cv2.circle(roi_color,
                               (ex+ew//2, ey+eh//2),
                               ew//2, (0,220,255), 2)
                    cv2.putText(roi_color, "OPEN",
                                (ex, max(0,ey-4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0,220,255), 1)

            # Drowsiness logic
            if face_found:
                if eyes_found == 0:
                    self.counter += 1
                    if self.counter >= thresh:
                        drowsy = True
                        play_alarm()
                        if self.counter == thresh:
                            self.drowsy_count += 1
                            ts = time.strftime("%H:%M:%S")
                            self.root.after(0, self._add_log,
                                            f"🔴 {ts} — DROWSY!")
                else:
                    self.counter = max(0, self.counter - 2)
                    if self.counter == 0:
                        stop_alarm()
            else:
                self.counter = max(0, self.counter - 1)
                if self.counter == 0:
                    stop_alarm()

            # HUD
            bg_c = (0,0,180) if drowsy else (0,130,50)
            txt  = "DROWSY - WAKE UP!" if drowsy else "ALERT - Eyes Open"
            cv2.rectangle(frame, (0,0), (w,44), (20,20,20), -1)
            cv2.putText(frame, "Driver Drowsiness Detector",
                        (8,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (220,220,220), 2)
            cv2.rectangle(frame, (0,h-44), (w,h), bg_c, -1)
            cv2.putText(frame, txt, (8,h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            self.root.after(0, self._update_ui,
                            frame.copy(), drowsy, eyes_found, bright)

        if self.cap:
            self.cap.release()

    def _update_ui(self, frame, drowsy, eyes_found, bright):
        # ── Fixed size resize — never grows ──────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        img   = img.resize((self.CAM_W, self.CAM_H), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.config(image=imgtk, text="")
        self.camera_label.image = imgtk

        if drowsy:
            self.status_label.config(
                text="😴  DROWSY — WAKE UP!", bg="#cc0000")
        else:
            self.status_label.config(
                text="✅  ALERT — Driver is Awake", bg="#006622")

        self.info_lbl.config(
            text=f"Eyes:{eyes_found} | Counter:{self.counter}/{self.sens_var.get()} | Bright:{bright} | FPS:{self.fps:.0f}")

        self.counter_val.config(text=str(self.counter))
        self.eyes_val.config(text=str(eyes_found))
        self.fps_val.config(text=f"{self.fps:.0f}")
        self.drowsy_val.config(text=str(self.drowsy_count))

    def _add_log(self, msg):
        self.log_entries.insert(0, msg)
        self.log_box.config(state="normal")
        self.log_box.delete(1.0, "end")
        self.log_box.insert("end", "\n".join(self.log_entries[:20]))
        self.log_box.config(state="disabled")

    def on_close(self):
        self.running = False
        stop_alarm()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = DrowsinessApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
