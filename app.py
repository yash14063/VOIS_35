from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import time
import math
import atexit # To close camera properly

app = Flask(__name__)

# ---------------- MEDIAPIPE CONFIG ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
TIP_IDS = [4, 8, 12, 16, 20]

# ---------------- GLOBAL STATE ----------------
camera_state = {
    "gesture_name": "Scanning...",
    "message": "Waiting...",
    "fingers": [],
    "emergency": False,
    "sos_countdown": 0
}

# ---------------- ROBUST CAMERA SETUP ----------------
# We do NOT open the camera globally anymore. 
# We open it only when the video feed starts.
cap = None 

def get_camera():
    global cap
    if cap is None or not cap.isOpened():
        # Try Index 0 with DirectShow (Faster on Windows)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # If 0 fails, try 1
        if not cap.isOpened():
            print("Camera 0 failed, trying Camera 1...")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            
    return cap

def release_camera():
    global cap
    if cap is not None:
        cap.release()

# Close camera when app stops
atexit.register(release_camera)

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_finger_state(hand_landmarks, results):
    fingers = []
    lm = hand_landmarks.landmark
    if results.multi_handedness:
        label = results.multi_handedness[0].classification[0].label
    else:
        label = "Right"

    # Thumb
    if label == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)

    # Fingers
    for id in range(1, 5):
        fingers.append(1 if lm[TIP_IDS[id]].y < lm[TIP_IDS[id] - 2].y else 0)

    return fingers

def detect_gesture(fingers, lm):
    state = tuple(fingers)
    
    # 1. SPECIAL GEOMETRIC GESTURES
    if calculate_distance(lm[4], lm[8]) < 0.05:
        return "OK Sign", "I am Okay"

    if fingers[1:] == [0,0,0,0] and lm[4].y > lm[3].y:
        return "Thumbs Down", "No"

    # 2. FINGER COMBINATION MAP
    gesture_map = {
        (0, 0, 0, 0, 0): ("Fist", "Holding..."),
        (1, 1, 1, 1, 1): ("Open Palm", "Hello"),
        (0, 1, 0, 0, 0): ("1 Finger", "One"),
        (0, 1, 1, 0, 0): ("Victory", "Two"),
        (0, 1, 1, 1, 0): ("3 Fingers", "Three"),
        (0, 1, 1, 1, 1): ("4 Fingers", "Four"),
        (1, 1, 1, 1, 0): ("4+Thumb", "Five"),
        (1, 0, 0, 0, 0): ("Thumb Up", "Yes"),
        (0, 0, 0, 0, 1): ("Pinky Up", "I need the Washroom"),
        (1, 1, 0, 0, 0): ("L-Shape", "Turn on Lights"),
        (1, 0, 0, 0, 1): ("Phone Hand", "Call the Doctor"),
        (0, 1, 0, 0, 1): ("Rock Sign", "Turn on TV"),
        (1, 1, 0, 0, 1): ("Spider-Man", "I Love You"),
        (1, 1, 1, 0, 0): ("Gun Shape", "I have Pain"),
        (0, 1, 1, 0, 1): ("Unique", "I am Hungry"),
        (1, 0, 1, 1, 1): ("OK Hand", "Perfect") 
    }
    
    return gesture_map.get(state, ("Unknown", "Scanning..."))

def generate_frames():
    fist_start_time = None
    SOS_HOLD_TIME = 2.0
    
    camera = get_camera() # Open camera only now
    
    while True:
        success, frame = camera.read()
        if not success:
            # If camera fails, try to reconnect
            camera = get_camera()
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        gesture_name = "Scanning..."
        message = "Scanning..."
        is_sos = False
        sos_timer = 0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_state(hand_landmarks, results)
            gesture_name, message = detect_gesture(fingers, hand_landmarks.landmark)

            if gesture_name == "Fist":
                if fist_start_time is None: fist_start_time = time.time()
                elapsed = time.time() - fist_start_time
                sos_timer = int(SOS_HOLD_TIME - elapsed) + 1
                if elapsed >= SOS_HOLD_TIME:
                    is_sos = True
                    message = "EMERGENCY SOS"
            else:
                fist_start_time = None

        camera_state.update({
            "gesture_name": gesture_name,
            "message": message,
            "emergency": is_sos,
            "sos_countdown": sos_timer
        })

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(camera_state)

if __name__ == '__main__':
    print("------------------------------------------------")
    print("✅ STARTING SERVER... Please wait.")
    print("📷 Go to this URL: http://127.0.0.1:5001")
    print("------------------------------------------------")
    # Change port to 5001 to avoid conflicts
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)