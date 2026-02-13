import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from hmmlearn import hmm


# config
DATA_PATH = Path("data/processed/normalized_samples.csv")
TEST_SIZE = 0.3
RANDOM_STATE = 42 # fixed random seed to ensure reproducibility of training results (could be any number)
STABILITY_WINDOW = 5 # for latency assessment


# load data
df = pd.read_csv(DATA_PATH)

# loading labels
labels_all = sorted(df["label"].unique())
gesture_labels = labels_all

# dismissal of unnecessary columns
non_feature_cols = ["sample_id", "user", "label", "timestamp"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values

label_to_int = {l: i for i, l in enumerate(labels_all)}
int_to_label = {i: l for l, i in label_to_int.items()}
y_int = np.array([label_to_int[l] for l in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_int,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_int
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# HMM training
hmms = {}

for label in gesture_labels:
    print("Training:", label)
    idx = y_train == label_to_int[label]
    X_class = X_train[idx]

    model = hmm.GaussianHMM(
        n_components=4,
        covariance_type="full",
        n_iter=100,
        random_state=RANDOM_STATE,
    )

    model.fit(X_class)
    hmms[label] = model


# HMM prediction
def predict_hmm(sample):
    best_label = None
    best_score = -np.inf

    for label, model in hmms.items():
        try:
            score = model.score(sample.reshape(1, -1))
        except:
            score = -np.inf

        if score > best_score:
            best_score = score
            best_label = label

    return best_label


# rule-based classification
def finger_extended(tip, pip, lm):
    return lm[tip][1] < lm[pip][1]

def thumb_extended(lm):
    return abs(lm[4][0] - lm[2][0]) > 0.04

def classify_rule(sample):
    lm = sample.reshape(21, 3)

    index_ext = finger_extended(8, 6, lm)
    middle_ext = finger_extended(12, 10, lm)
    ring_ext = finger_extended(16, 14, lm)
    little_ext = finger_extended(20, 18, lm)
    thumb_ext = thumb_extended(lm)

    wrist_y = lm[0][1]
    thumb_y = lm[4][1]

    # rules
    if index_ext and middle_ext and ring_ext and little_ext and thumb_ext:
        return "forward"

    if not index_ext and not middle_ext and not ring_ext and not little_ext:
        if not thumb_ext:
            return "back"

    if thumb_ext and not index_ext and not middle_ext and not ring_ext and not little_ext:
        if thumb_y < wrist_y:
            return "takeoff"
        if thumb_y > wrist_y:
            return "land"

    if index_ext and little_ext and not middle_ext and not ring_ext and not thumb_ext:
        return "up"

    if index_ext and little_ext and thumb_ext and not middle_ext and not ring_ext:
        return "down"

    if index_ext and not middle_ext and not ring_ext and not little_ext:
        return "left"

    if little_ext and not index_ext and not middle_ext and not ring_ext:
        return "right"

    return "noise"


# evaluation
hmm_preds = []
rule_preds = []

for x in X_test:
    hmm_preds.append(predict_hmm(x))
    rule_preds.append(classify_rule(x))

hmm_preds = np.array(hmm_preds)
rule_preds = np.array(rule_preds)

y_test_labels = np.array([int_to_label[i] for i in y_test])


# evaluation metrics
def evaluate(name, preds):

    acc = accuracy_score(y_test_labels, preds)
    f1 = f1_score(y_test_labels, preds, average="macro")

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Macro F1:", f1)

    cm = confusion_matrix(y_test_labels, preds, labels=labels_all)

    plt.figure(figsize=(8,6))
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.xticks(range(len(labels_all)), labels_all, rotation=45)
    plt.yticks(range(len(labels_all)), labels_all)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

evaluate("HMM", hmm_preds)
evaluate("Rule-based", rule_preds)


# false positive rate test
noise_idx = y_test_labels == "noise"

hmm_fp = np.sum((hmm_preds != "noise") & noise_idx)
rule_fp = np.sum((rule_preds != "noise") & noise_idx)

noise_count = np.sum(noise_idx)

print("\n=== FALSE POSITIVE RATE ===")
print("HMM FPR:", hmm_fp / noise_count)
print("Rule FPR:", rule_fp / noise_count)
