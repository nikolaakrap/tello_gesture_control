import cv2
import numpy as np
import joblib
from collections import deque
from pathlib import Path
import mediapipe as mp
from djitellopy import Tello
import time
import csv


# config
MODEL_DIR = Path("data/models")
LOG_PATH = Path("data/latency_trials.csv")

SEQ_LEN = 20
RC_SPEED = 25
VELOCITY_THRESHOLD = 0.1   # lower threshold
LOOP_DELAY = 0.05


# load HMM models
hmms = {}
for pkl in MODEL_DIR.glob("hmm_*.pkl"):
    label = pkl.stem.replace("hmm_", "")
    if label != "noise":
        hmms[label] = joblib.load(pkl)

# csv init
if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id",
            "gesture",
            "decision_latency_ms",
            "actuation_latency_ms",
            "end_to_end_latency_ms"
        ])


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


# HMM prediction
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


# RC commands
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
    trial_active = False
    trial_id = 0

    t_detect = None
    t_cmd = None
    rc_sent = False

    print("\nPress SPACE to start latency trial.")
    print("Press q to quit.\n")

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

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not is_flying:
                tello.takeoff()
                is_flying = True
                time.sleep(2)

            trial_active = True
            rc_sent = False
            t_detect = None
            t_cmd = None
            print(f"\nTrial {trial_id} started.")

        if key == ord("q"):
            break

        # gesture detection
        if trial_active and detected_gesture and not rc_sent:

            t_detect = time.perf_counter()

            lr, fb, ud, yaw = get_rc_command(detected_gesture)

            t_cmd = time.perf_counter()
            tello.send_rc_control(lr, fb, ud, yaw)

            rc_sent = True
            print(f"Gesture detected: {detected_gesture}")


        # latency measurement
        if trial_active and rc_sent and t_cmd is not None:

            try:
                velocity = (
                    abs(tello.get_speed_x()) +
                    abs(tello.get_speed_y()) +
                    abs(tello.get_speed_z())
                )
            except:
                velocity = 0

            if velocity > VELOCITY_THRESHOLD:

                t_move = time.perf_counter()

                decision_latency = (t_cmd - t_detect) * 1000
                actuation_latency = (t_move - t_cmd) * 1000
                end_to_end_latency = (t_move - t_detect) * 1000

                print(f"Decision latency: {round(decision_latency,2)} ms")
                print(f"Actuation latency: {round(actuation_latency,2)} ms")
                print(f"End-to-end latency: {round(end_to_end_latency,2)} ms\n")

                with open(LOG_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        trial_id,
                        detected_gesture,
                        round(decision_latency, 2),
                        round(actuation_latency, 2),
                        round(end_to_end_latency, 2)
                    ])

                tello.send_rc_control(0,0,0,0)

                trial_active = False
                trial_id += 1

        cv2.putText(frame,
                    f"Trial: {trial_id}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        cv2.imshow("07 - Latency Test", frame)

        elapsed = time.perf_counter() - loop_start
        if elapsed < LOOP_DELAY:
            time.sleep(LOOP_DELAY - elapsed)

    if is_flying:
        tello.send_rc_control(0,0,0,0)
        tello.land()

    cap.release()
    cv2.destroyAllWindows()
    tello.end()

    print("\nLatency test finished.")
    print("Results saved to:", LOG_PATH)


if __name__ == "__main__":
    main()
