# app/app.py
import os
import json
import time
import threading
from collections import deque

import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

# -----------------------------
# Paths (relative to this file)
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "sign_model.h5")
CLASS_INDICES_PATH = os.path.join(ROOT_DIR, "models", "class_indices.json")

# -----------------------------
# Inference settings
# -----------------------------
IMG_SIZE = (64, 64)      # must match training
CONF_THRESHOLD = 0.50     # only speak/show when fairly confident
SMOOTHING_WINDOW = 7      # majority voting over last N frames

# -----------------------------
# Speech engine (non-blocking)
# -----------------------------
class SpeechEngine:
    def __init__(self, enabled=True, rate=175):
        self.enabled = enabled
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self._lock = threading.Lock()
        self._last_spoken = ""
        self._thread = None

    def _speak_thread(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass

    def speak(self, text):
        if not self.enabled:
            return
        # Avoid spamming the same letter repeatedly
        with self._lock:
            if text == self._last_spoken:
                return
            self._last_spoken = text
        # Run in background
        t = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        t.start()
        self._thread = t

    def toggle(self):
        self.enabled = not self.enabled
        # Reset last spoken so it can speak current letter again after toggle
        with self._lock:
            self._last_spoken = ""
        return self.enabled


# -----------------------------
# Utilities
# -----------------------------
def load_class_mapping(path):
    """
    class_indices.json is a dict like {"A":0,"B":1,...} (label -> index)
    We need index->label for decoding predictions.
    """
    with open(path, "r") as f:
        label_to_idx = json.load(f)
    # Some tools save as {label: idx}, others as {idx: label}; handle both
    # Normalize to idx->label dict
    try:
        # if keys are strings of letters (A,B,...) then invert
        if all(isinstance(k, str) for k in label_to_idx.keys()):
            idx_to_label = {v: k for k, v in label_to_idx.items()}
        else:
            # keys might be numeric strings "0","1",...
            idx_to_label = {int(k): v for k, v in label_to_idx.items()}
    except Exception:
        # Fallback: try to coerce whatever is there
        idx_to_label = {}
        for k, v in label_to_idx.items():
            try:
                idx_to_label[int(v)] = k
            except Exception:
                pass
    return idx_to_label

def preprocess_frame(frame_bgr, target_size=(64, 64)):
    """
    Center-crop to square, resize, scale to [0,1]
    """
    h, w = frame_bgr.shape[:2]
    # center-crop square
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame_bgr[y0:y0+side, x0:x0+side]

    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    # Convert BGR->RGB because Keras usually expects RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype("float32") / 255.0
    # shape: (1, H, W, 3)
    return np.expand_dims(arr, axis=0)

def draw_overlay(frame, text, conf, fps, speech_on):
    """
    Draw prediction, confidence, FPS, and help text.
    """
    overlay = frame.copy()
    H, W = frame.shape[:2]

    # semi-transparent banner
    cv2.rectangle(overlay, (0, 0), (W, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # Main prediction
    cv2.putText(frame, f"Pred: {text}  (conf: {conf:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # FPS & Speech
    cv2.putText(frame, f"FPS: {fps:.1f}   Speech: {'ON' if speech_on else 'OFF'}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    # Help bar at bottom
    cv2.rectangle(frame, (0, H-35), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, "Q: Quit   S: Toggle Speech   C: Switch Camera",
                (20, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    return frame


# -----------------------------
# Main
# -----------------------------
def main(camera_index=0):
    # Validate files
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(f"class_indices.json not found at: {CLASS_INDICES_PATH}")

    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded.")

    idx_to_label = load_class_mapping(CLASS_INDICES_PATH)
    num_classes = len(idx_to_label)
    if num_classes == 0:
        raise RuntimeError("No classes found in class_indices.json")

    # Prepare webcam
    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {camera_index}")

    # FPS calc
    prev_time = time.time()
    fps = 0.0

    # Smoothing predictions
    recent_preds = deque(maxlen=SMOOTHING_WINDOW)

    # Speech engine
    speaker = SpeechEngine(enabled=False)  # start with OFF; toggle with 'S'
    print("[INFO] Webcam started. Press 'S' to toggle speech, 'Q' to quit.")

    current_cam = camera_index

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("[WARN] Failed to read frame. Trying again...")
                time.sleep(0.02)
                continue

            # Preprocess
            x = preprocess_frame(frame, IMG_SIZE)
            # Predict
            probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx]) if probs.size else 0.0
            pred_label = idx_to_label.get(pred_idx, "?")

            # Majority vote smoothing (only count confident frames)
            if conf >= CONF_THRESHOLD:
                recent_preds.append(pred_label)
            # Pick most common in window
            if recent_preds:
                values, counts = np.unique(recent_preds, return_counts=True)
                smooth_label = values[np.argmax(counts)]
                smooth_conf = conf
            else:
                smooth_label = "-"
                smooth_conf = 0.0

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else fps

            # Draw overlay
            vis = draw_overlay(frame, smooth_label, smooth_conf, fps, speaker.enabled)
            cv2.imshow("Sign Language - Live Inference", vis)

            # Speech (speak only when confidence high and label not "-")
            if speaker.enabled and smooth_label not in ["-", "?"] and conf >= CONF_THRESHOLD:
                speaker.speak(smooth_label)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S')):
                enabled = speaker.toggle()
                print(f"[INFO] Speech {'ENABLED' if enabled else 'DISABLED'}")
            elif key in (ord('c'), ord('C')):
                # Switch camera (cycle 0 -> 1 -> 2 -> 0)
                next_cam = (current_cam + 1) % 3
                print(f"[INFO] Switching camera to index {next_cam} ...")
                cam.release()
                time.sleep(0.2)
                cam = cv2.VideoCapture(next_cam, cv2.CAP_DSHOW)
                if cam.isOpened():
                    current_cam = next_cam
                else:
                    print(f"[WARN] Camera index {next_cam} not available. Reverting.")
                    cam.release()
                    cam = cv2.VideoCapture(current_cam, cv2.CAP_DSHOW)

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Closed.")

if __name__ == "__main__":
    # If you want to start with camera index 1, change here to main(1)
    main(0)
