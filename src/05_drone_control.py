import cv2
import numpy as np
import joblib
from collections import deque
from pathlib import Path
import mediapipe as mp
from djitellopy import Tello
import time


# config
MODEL_DIR = Path("data/models")

SEQ_LEN = 20
RC_SPEED = 25
LOOP_DELAY = 0.05
VELOCITY_THRESHOLD = 1  # cm/s


# load trained HMMs
hmms = {}
for pkl in MODEL_DIR.glob("hmm_*.pkl"):
    label = pkl.stem.replace("hmm_", "")
    if label != "noise":
        hmms[label] = joblib.load(pkl)


# mediapipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


# normalization
def normalize_hand(coords):
    wrist = coords[0]
    centered = coords - wrist
    scale = np.linalg.norm(coords[9] - wrist)
    if scale < 1e-6:
        scale = 1.0
    return centered / scale


def extract_features(result):
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
    coords = normalize_hand(coords)
    return coords.reshape(-1)


# gesture scoring & prediction
def predict_hmm(sequence):
    best_label = None
    best_score = -np.inf
    for label, model in hmms.items():
        try:
            score = model.score(sequence)
        except:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


# RC commands that will be sent to the drone
def get_rc_command(gesture):
    match gesture:
        case "right":
            return (RC_SPEED, 0, 0, 0)
        case "left":
            return (-RC_SPEED, 0, 0, 0)
        case "forward":
            return (0, RC_SPEED, 0, 0)
        case "back":
            return (0, -RC_SPEED, 0, 0)
        case "up":
            return (0, 0, RC_SPEED, 0)
        case "down":
            return (0, 0, -RC_SPEED, 0)
        case _:
            return (0, 0, 0, 0)


def main():

    tello = Tello()
    tello.connect()
    print("Battery:", tello.get_battery(), "%")

    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQ_LEN)

    is_flying = False

    print("\nPress q to quit\n")

    while True:

        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        feat = extract_features(result)

        if feat is not None:
            buffer.append(feat)
        else:
            buffer.clear()

        detected_gesture = None

        if len(buffer) == SEQ_LEN:
            seq = np.array(buffer, dtype=np.float32)
            detected_gesture = predict_hmm(seq)


        # drone control logic
        if detected_gesture is None:
            if is_flying:
                tello.send_rc_control(0, 0, 0, 0)
        else:
            # takeoff only available if drone is not flying
            if detected_gesture == "takeoff" and not is_flying:
                tello.takeoff()
                is_flying = True
            # land only available if drone is flying
            elif detected_gesture == "land" and is_flying:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                is_flying = False
            elif is_flying:
                lr, fb, ud, yaw = get_rc_command(detected_gesture)
                tello.send_rc_control(lr, fb, ud, yaw)

        # ui
        state_text = detected_gesture if detected_gesture else "HOVER"

        cv2.putText(frame,
                    f"STATE: {state_text}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        cv2.imshow("05 - Drone Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if is_flying:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
            break

        elapsed = time.perf_counter() - loop_start
        if elapsed < LOOP_DELAY:
            time.sleep(LOOP_DELAY - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    tello.end()


if __name__ == "__main__":
    main()
