import cv2
import numpy as np
import joblib
from collections import deque
from pathlib import Path
import mediapipe as mp


# config
MODEL_DIR = Path("data/models")
SEQ_LEN = 20
SCORE_MARGIN = 150.0
DRAW_LANDMARKS = True


# load trained HMMs
hmms = {}
for pkl in MODEL_DIR.glob("hmm_*.pkl"):
    label = pkl.stem.replace("hmm_", "")
    hmms[label] = joblib.load(pkl)
    print(f"Loaded HMM: {label}")

if "noise" not in hmms:
    raise RuntimeError("Noise HMM missing – required!")

gesture_labels = [l for l in hmms.keys() if l != "noise"]
print("Available gestures:", gesture_labels)


# mediapipe init
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

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


# confidence computation
def compute_confidence(scores):
    labels = list(scores.keys())
    vals = np.array([scores[l] for l in labels], dtype=np.float64)

    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        return {l: 0 for l in labels}

    min_finite = np.min(vals[finite_mask])
    vals = np.where(finite_mask, vals, min_finite - 1000.0)

    vals = vals - np.min(vals)
    mx = np.max(vals)
    if mx <= 1e-12:
        return {l: 0 for l in labels}

    vals = vals / mx
    conf = np.clip(np.round(vals * 100.0), 0, 100).astype(int)
    return {l: int(c) for l, c in zip(labels, conf)}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access camera")

    buffer = deque(maxlen=SEQ_LEN)

    current_state = "HOVER"
    active_gesture = None

    print("\nControls:")
    print("  q : quit")
    print("  d : toggle landmarks\n")

    global DRAW_LANDMARKS

    while True:
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

        detected_label = "—"
        scores = {}
        confidences = {}

        # HMM scoring
        if len(buffer) == SEQ_LEN:
            seq = np.array(buffer, dtype=np.float32)

            for label, model in hmms.items():
                try:
                    scores[label] = model.score(seq)
                except Exception:
                    scores[label] = -np.inf

            confidences = compute_confidence(scores)

            # nosie-dependent decision logic
            noise_score = scores.get("noise", -np.inf)

            # best not-NOISE gesture
            best_label = None
            best_score = -np.inf

            for lbl in gesture_labels:
                if scores[lbl] > best_score:
                    best_score = scores[lbl]
                    best_label = lbl

            if best_label is not None:
                if best_score >= noise_score + SCORE_MARGIN:
                    detected_label = best_label

        # event logic
        if detected_label == "—":
            if current_state != "HOVER":
                print("→ HOVER")
            current_state = "HOVER"
            active_gesture = None
        else:
            if active_gesture != detected_label:
                print(f"→ GESTURE: {detected_label}")
            current_state = "GESTURE"
            active_gesture = detected_label

        # draw landmarks
        if DRAW_LANDMARKS and result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
            )

        # ui
        y = 25
        for lbl, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            conf = confidences.get(lbl, 0)
            cv2.putText(
                frame,
                f"{lbl:12s}: {sc:8.1f} | {conf:3d}%",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            y += 18

        state_text = "HOVER" if current_state == "HOVER" else active_gesture

        cv2.putText(
            frame,
            f"STATE: {state_text}",
            (10, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if current_state == "GESTURE" else (0, 200, 255),
            2,
        )

        cv2.imshow("04 - Realtime HMM Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            DRAW_LANDMARKS = not DRAW_LANDMARKS

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
