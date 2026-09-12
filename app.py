from abc import ABC, abstractmethod
import os

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))




COLOR_INFO = {
    "Red":     {"hex": "#e53935", "bgr": (53, 57, 229)},
    "Orange":  {"hex": "#fb8c00", "bgr": (0, 140, 251)},
    "Yellow":  {"hex": "#fdd835", "bgr": (53, 216, 253)},
    "Green":   {"hex": "#43a047", "bgr": (71, 160, 67)},
    "Blue":    {"hex": "#1e88e5", "bgr": (229, 136, 30)},
    "Purple":  {"hex": "#8e24aa", "bgr": (170, 36, 142)},
    "Pink":    {"hex": "#ec407a", "bgr": (122, 64, 236)},
    "Black":   {"hex": "#212121", "bgr": (33, 33, 33)},
    "White":   {"hex": "#fafafa", "bgr": (250, 250, 250)},
    "Gray":    {"hex": "#9e9e9e", "bgr": (158, 158, 158)},
    "Unknown": {"hex": "#9e9e9e", "bgr": (158, 158, 158)},
}




class ColorDetectionStrategy(ABC):
    @abstractmethod
    def detect_color(self, frame: np.ndarray) -> tuple:
       
        pass


class SweaterColorDetector(ColorDetectionStrategy):
    
   

    MIN_AREA = 1500        
    MIN_SAMPLE = 600       
    BORDER_MARGIN = 8      
    SAT_MIN = 80           

    KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    
    SKIN_LOWER = np.array([0, 40, 60], dtype=np.uint8)
    SKIN_UPPER = np.array([25, 170, 255], dtype=np.uint8)

    def _colorful_mask(self, hsv: np.ndarray) -> np.ndarray:
        
        mask = cv2.inRange(hsv, np.array([0, self.SAT_MIN, 40], dtype=np.uint8),
                           np.array([180, 255, 255], dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.KERNEL, iterations=2)
        return mask

    def _touches_border(self, x, y, w, h, fw, fh) -> bool:
       
        return (x <= self.BORDER_MARGIN or y <= self.BORDER_MARGIN or
                x + w >= fw - self.BORDER_MARGIN or
                y + h >= fh - self.BORDER_MARGIN)

    @staticmethod
    def _classify(h_mean: float, s_mean: float, v_mean: float) -> str:
        
        if v_mean < 45:
            return "Black"
        if s_mean < 40:
            return "White" if v_mean > 170 else "Gray"
        if h_mean <= 10 or h_mean >= 170:
            return "Red"
        if h_mean <= 25:
            return "Orange"
        if h_mean <= 35:
            return "Yellow"
        if h_mean <= 85:
            return "Green"
        if h_mean <= 125:
            return "Blue"
        if h_mean <= 155:
            return "Purple"
        return "Pink"

    def detect_color(self, frame: np.ndarray) -> tuple:
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colorful = self._colorful_mask(hsv)
        skin = cv2.inRange(hsv, self.SKIN_LOWER, self.SKIN_UPPER)

        contours, _ = cv2.findContours(colorful, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.MIN_AREA]
        if not contours:
            return "Unknown", None

        
        contours.sort(key=cv2.contourArea, reverse=True)

        
        ordered = sorted(contours,
                         key=lambda c: self._touches_border(*cv2.boundingRect(c), fw, fh))

        for contour in ordered:
            x, y, w, h = cv2.boundingRect(contour)

            
            inside = np.zeros((fh, fw), dtype=np.uint8)
            cv2.drawContours(inside, [contour], -1, 255, thickness=-1)
            inside = cv2.bitwise_and(inside, colorful)

            
            sample = cv2.bitwise_and(inside, cv2.bitwise_not(skin))
            if cv2.countNonZero(sample) < self.MIN_SAMPLE:
                sample = inside  
            if cv2.countNonZero(sample) < self.MIN_SAMPLE:
                continue  

            
            hs = hsv[:, :, 0][sample > 0].astype(np.float32) * 2.0
            ss = hsv[:, :, 1][sample > 0].astype(np.float32)
            vs = hsv[:, :, 2][sample > 0].astype(np.float32)

            rad = np.deg2rad(hs)
            h_mean = float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)),
                                                 np.mean(np.cos(rad)))) % 360.0) / 2.0
            color_name = self._classify(h_mean, float(np.mean(ss)), float(np.mean(vs)))
            return color_name, (x, y, x + w, y + h)

        return "Unknown", None




current_color = "Unknown"


def get_color_payload():
    info = COLOR_INFO.get(current_color, COLOR_INFO["Unknown"])
    return {"name": current_color, "hex": info["hex"]}




class CameraStream:
    def __init__(self, detector: ColorDetectionStrategy, camera_index: int = 0):
        self.detector = detector
        self.camera_index = camera_index

    def generate_frames(self):
        global current_color

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: cannot open camera.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                color_name, box = self.detector.detect_color(frame)
                current_color = color_name

                if box:
                    x1, y1, x2, y2 = box
                  
                    bgr = COLOR_INFO[color_name]["bgr"]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, -1)
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, color_name, (x1, max(y1 - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        finally:
            cap.release()




detector = SweaterColorDetector()
stream = CameraStream(detector=detector, camera_index=0)


@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Template error: {e}", 500


@app.route("/video_feed")
def video_feed():
    return Response(
        stream.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/color")
def color_status():
    return jsonify(get_color_payload())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)