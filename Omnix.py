import cv2
import pyautogui
import math
import time
import urllib.request
import os
import numpy as np
import tkinter as tk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configure PyAutoGUI
pyautogui.FAILSAFE = True  # Move mouse to the corner of the screen to stop the program
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# One Euro Filter for smooth cursor movement
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return x
        
        dx = (x - self.x_prev) / t_e
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class MovingAverage:
    def __init__(self, size=8):
        self.window = []
        self.size = size
    def update(self, val):
        self.window.append(val)
        if len(self.window) > self.size:
            self.window.pop(0)
        return sum(self.window) / len(self.window)

def get_landmark_coords(face_landmarks, index, frame_width, frame_height):
    x = int(face_landmarks[index].x * frame_width)
    y = int(face_landmarks[index].y * frame_height)
    return (x, y)

def calculate_ear(eye_points):
    if len(eye_points) < 6: return 0.0
    A = math.dist(eye_points[1], eye_points[5])
    B = math.dist(eye_points[2], eye_points[4])
    C = math.dist(eye_points[0], eye_points[3])
    if C == 0: return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def download_model():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        print(f"Downloading model {model_path} it may take a minute...")
        urllib.request.urlretrieve(url, model_path)
    return model_path

def draw_vector_emoji(frame, expression, x_center, y_center, size=40, color=(0, 255, 0)):
    # Black background
    cv2.rectangle(frame, (x_center - size - 15, y_center - size - 15), 
                  (x_center + size + 15, y_center + size + 15), (0, 0, 0), -1)
    
    # Face outline
    cv2.circle(frame, (x_center, y_center), size, color, 3)
    
    x, y = x_center, y_center
    
    if expression == "happiness":
        cv2.ellipse(frame, (x, y + 5), (20, 15), 0, 0, 180, color, 3) # Smile
        cv2.ellipse(frame, (x - 15, y - 10), (8, 5), 0, 180, 360, color, 3) 
        cv2.ellipse(frame, (x + 15, y - 10), (8, 5), 0, 180, 360, color, 3) 
    elif expression == "sadness":
        cv2.ellipse(frame, (x, y + 20), (20, 12), 0, 180, 360, color, 3) # Frown
        cv2.line(frame, (x - 20, y - 15), (x - 8, y - 10), color, 3) 
        cv2.line(frame, (x + 8, y - 10), (x + 20, y - 15), color, 3) 
        cv2.circle(frame, (x - 15, y - 5), 4, color, 2) 
        cv2.circle(frame, (x + 15, y - 5), 4, color, 2) 
    elif expression == "anger":
        cv2.line(frame, (x - 15, y + 15), (x + 15, y + 15), color, 3) 
        cv2.line(frame, (x - 22, y - 18), (x - 8, y - 5), color, 4) # \
        cv2.line(frame, (x + 8, y - 5), (x + 22, y - 18), color, 4) # /
        cv2.circle(frame, (x - 15, y - 5), 5, color, -1) 
        cv2.circle(frame, (x + 15, y - 5), 5, color, -1)
    elif expression in ["surprise", "shock"]:
        cv2.circle(frame, (x, y + 15), 10, color, 3) 
        cv2.circle(frame, (x - 15, y - 5), 6, color, 3) 
        cv2.circle(frame, (x + 15, y - 5), 6, color, 3)
        cv2.ellipse(frame, (x - 15, y - 25), (8, 4), 0, 180, 360, color, 3) 
        cv2.ellipse(frame, (x + 15, y - 25), (8, 4), 0, 180, 360, color, 3)
    elif expression == "skepticism":
        cv2.line(frame, (x - 15, y + 15), (x + 15, y + 10), color, 3) 
        cv2.ellipse(frame, (x - 15, y - 20), (8, 4), 0, 180, 360, color, 3) # High
        cv2.circle(frame, (x - 15, y - 10), 4, color, 3) 
        cv2.line(frame, (x + 8, y - 10), (x + 20, y - 6), color, 3) # Low
        cv2.line(frame, (x + 10, y - 5), (x + 20, y - 5), color, 3) # Squint
    elif expression == "boredom":
        cv2.line(frame, (x - 15, y + 15), (x + 15, y + 15), color, 3) 
        cv2.line(frame, (x - 22, y - 5), (x - 8, y - 5), color, 3) 
        cv2.line(frame, (x + 8, y - 5), (x + 22, y - 5), color, 3) 
    else: # neutral
        cv2.line(frame, (x - 15, y + 15), (x + 15, y + 15), color, 3) 
        cv2.circle(frame, (x - 15, y - 10), 4, color, 3) 
        cv2.circle(frame, (x + 15, y - 10), 4, color, 3)

def main():
    model_path = download_model()
    
    # Initialize MediaPipe Task
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
        
    detector = vision.FaceLandmarker.create_from_options(options)

    cam = cv2.VideoCapture(0)
    
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    
    t_init = time.time()
    # Much stronger filter parameters for maximum stability
    filter_x = OneEuroFilter(t_init, 0.0, min_cutoff=0.01, beta=0.001)
    filter_y = OneEuroFilter(t_init, 0.0, min_cutoff=0.01, beta=0.001)
    
    ma_x = MovingAverage(size=8)
    ma_y = MovingAverage(size=8)
    
    # Initialize click-through desktop Eye icon overlay
    eye_root = tk.Tk()
    eye_root.overrideredirect(True)
    eye_root.attributes('-topmost', True)
    eye_root.attributes('-transparentcolor', 'white')
    eye_root.attributes('-disabled', True) 
    
    eye_canvas = tk.Canvas(eye_root, width=60, height=60, bg='white', highlightthickness=0)
    eye_canvas.pack()
    eye_canvas.create_oval(5, 22, 55, 38, outline='black', width=2, fill='#FFF')
    eye_canvas.create_oval(22, 22, 38, 38, outline='black', width=2, fill='#1E90FF')
    eye_canvas.create_oval(27, 27, 33, 33, outline='', fill='black')
    eye_canvas.create_oval(25, 24, 28, 27, outline='', fill='white')
    eye_root.geometry("60x60+0+0")
    eye_root.update()

    last_click_time = time.time()
    BLINK_THRESHOLD = 0.2
    COOLDOWN = 1.0  
    
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        avg_ear = 0.0
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # Blink Detection
            left_eye_points = [get_landmark_coords(landmarks, i, w, h) for i in LEFT_EYE]
            right_eye_points = [get_landmark_coords(landmarks, i, w, h) for i in RIGHT_EYE]
            
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < BLINK_THRESHOLD:
                if time.time() - last_click_time > COOLDOWN:
                    try:
                        pyautogui.click()
                        print("Blink Detected -> Clicked!")
                    except:
                        pass
                    last_click_time = time.time()
                    
            # Gaze Mapping: Average both eyes and use RIGID eye corners for stability
            # Right Eye
            r_iris = landmarks[RIGHT_IRIS_CENTER]
            r_x_center = (landmarks[362].x + landmarks[263].x) / 2.0
            r_y_center = (landmarks[362].y + landmarks[263].y) / 2.0
            r_width = abs(landmarks[362].x - landmarks[263].x) + 1e-6
            r_norm_x = (r_iris.x - r_x_center) / r_width
            r_norm_y = (r_iris.y - r_y_center) / r_width
            
            # Left Eye
            l_iris = landmarks[LEFT_IRIS_CENTER]
            l_x_center = (landmarks[33].x + landmarks[133].x) / 2.0
            l_y_center = (landmarks[33].y + landmarks[133].y) / 2.0
            l_width = abs(landmarks[33].x - landmarks[133].x) + 1e-6
            l_norm_x = (l_iris.x - l_x_center) / l_width
            l_norm_y = (l_iris.y - l_y_center) / l_width
            
            # Average both eyes
            raw_norm_x = (r_norm_x + l_norm_x) / 2.0
            raw_norm_y = (r_norm_y + l_norm_y) / 2.0
            
            # Smooth coordinate flutter using Moving Average
            norm_x = ma_x.update(raw_norm_x)
            norm_y = ma_y.update(raw_norm_y)

            # Tuning sensitivities based on relative offset from eye center
            # Pupil moves within approx +/- 0.15 ratio from center
            sens_x_min, sens_x_max = -0.15, 0.15
            sens_y_min, sens_y_max = -0.15, 0.15
            
            screen_x_ratio = (norm_x - sens_x_min) / (sens_x_max - sens_x_min)
            screen_y_ratio = (norm_y - sens_y_min) / (sens_y_max - sens_y_min)
            
            screen_x_ratio = max(0.0, min(1.0, screen_x_ratio))
            screen_y_ratio = max(0.0, min(1.0, screen_y_ratio))
            
            raw_screen_x = screen_x_ratio * SCREEN_WIDTH
            raw_screen_y = screen_y_ratio * SCREEN_HEIGHT
            
            t_now = time.time()
            smooth_x = filter_x(t_now, raw_screen_x)
            smooth_y = filter_y(t_now, raw_screen_y)
            
            try:
                # Clamp safely within screen mapping to avoid Tkinter geometry crashes
                clamped_x = max(30, min(SCREEN_WIDTH - 30, smooth_x))
                clamped_y = max(30, min(SCREEN_HEIGHT - 30, smooth_y))
                
                pyautogui.moveTo(clamped_x, clamped_y)
                eye_root.geometry(f"60x60+{int(clamped_x)-30}+{int(clamped_y)-30}")
            except Exception as e:
                pass 
                
            cx, cy = get_landmark_coords(landmarks, RIGHT_IRIS_CENTER, w, h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            
            for pt in left_eye_points + right_eye_points:
                cv2.circle(frame, pt, 1, (255, 0, 0), -1)
                
            # Advanced Expression Detection
            face_w = math.dist((landmarks[234].x, landmarks[234].y), (landmarks[454].x, landmarks[454].y)) + 1e-6
            face_h = math.dist((landmarks[10].x, landmarks[10].y), (landmarks[152].x, landmarks[152].y)) + 1e-6
            mouth_w = math.dist((landmarks[61].x, landmarks[61].y), (landmarks[291].x, landmarks[291].y))
            mouth_h = math.dist((landmarks[13].x, landmarks[13].y), (landmarks[14].x, landmarks[14].y))
            
            mouth_center_y = (landmarks[13].y + landmarks[14].y) / 2.0
            corner_y = (landmarks[61].y + landmarks[291].y) / 2.0
            smile_curve = (mouth_center_y - corner_y) / face_h # Positive if smiling (corners go up/smaller Y)
            
            left_brow_h = (landmarks[159].y - landmarks[105].y) / face_h # Distance from eye to brow
            right_brow_h = (landmarks[386].y - landmarks[334].y) / face_h
            avg_brow_h = (left_brow_h + right_brow_h) / 2.0
            brow_asymmetry = abs(left_brow_h - right_brow_h)

            expression = "neutral"
            
            if avg_ear < 0.22:
                expression = "boredom"
            elif brow_asymmetry > 0.025:
                expression = "skepticism"
            elif mouth_h / face_h > 0.08:
                if avg_brow_h > 0.07:
                    expression = "surprise"
                else:
                    expression = "shock"
            elif smile_curve > 0.012 or mouth_w / face_w > 0.42:
                expression = "happiness"
            elif smile_curve < -0.015:
                expression = "sadness"
                if avg_brow_h < 0.045:
                    expression = "anger"
            elif avg_brow_h < 0.045:
                expression = "anger"

            draw_vector_emoji(frame, expression, int(w - 70), 70, size=40, color=(0, 255, 0))
                
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "GazeFlow Tracking Active", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.imshow("GazeFlow Camera", frame)
        try:
            eye_root.update()
        except:
            pass
            
        if cv2.waitKey(1) & 0xFF == 27: # Esc to exit
            break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
