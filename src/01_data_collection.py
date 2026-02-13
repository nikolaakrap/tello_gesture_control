import cv2
import csv
import time
from pathlib import Path
import mediapipe as mp


# config
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# user labels (add more if necessary)
USERS = ["user_01", "user_02", "user_03"]

# gesture labels (add more if necessary)
LABELS = [
    "takeoff",
    "land",
    "up",
    "down",
    "left",
    "right",
    "forward",
    "back",
    "noise"
]


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


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access camera")

    # saving a new CSV for every session
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = DATA_DIR / f"hand_samples_{run_ts}.csv"

    user_idx = 0
    label_idx = 0
    sample_id = 0
    start_time = time.time()

    # CSV initialization
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_id", "user", "label", "timestamp"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

    print(f"\nSaving samples to: {csv_path}")
    print("\nControls:")
    print("  l : next label")
    print("  u : change user")
    print("  g : save current frame")
    print("  q : quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = None

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for p in hand_lm.landmark:
                landmarks.extend([p.x, p.y, p.z])

        # ui
        label_color = (0, 255, 0)

        cv2.putText(
            frame,
            f"User:  {USERS[user_idx]}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Label: {LABELS[label_idx]}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2,
        )

        cv2.imshow("01 - Hand Gesture Data Collection", frame)


        # key input handling
        key = cv2.waitKey(1) & 0xFF

        # save & exit
        if key == ord("q"):
            break

        # change gesture label
        elif key == ord("l"):
            label_idx = (label_idx + 1) % len(LABELS)
            print(f"Label -> {LABELS[label_idx]}")

        # change user label
        elif key == ord("u"):
            user_idx = (user_idx + 1) % len(USERS)
            print(f"User -> {USERS[user_idx]}")

        # save frame to CSV
        elif key == ord("g"):
            if landmarks is None:
                print("No hand detected – sample not saved.")
                continue

            timestamp = time.time() - start_time

            row = [
                sample_id,
                USERS[user_idx],
                LABELS[label_idx],
                round(timestamp, 4),
            ] + landmarks

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(
                f"Saved sample {sample_id:04d} | "
                f"user={USERS[user_idx]} | "
                f"label={LABELS[label_idx]} | "
                f"t={timestamp:.2f}s"
            )

            sample_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
