import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
import joblib


# config
DATA_CSV = Path("data/processed/normalized_samples.csv")
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = 4 # number of hidden states per HMM (phases of the gesture)
COV_TYPE = "full"
VAR_FLOOR = 1e-4 # minimal covariance value (prevents numerical instability)

RANDOM_STATE = 42 # fixed random seed to ensure reproducibility of training results (could be any number)


def main():
    if not DATA_CSV.exists():
        raise RuntimeError("Normalized dataset not found.")

    df = pd.read_csv(DATA_CSV)

    feature_cols = [c for c in df.columns if c.startswith("f")]
    X_all = df[feature_cols].values
    labels = df["label"].values

    print(f"Loaded dataset:")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(feature_cols)}")

    unique_labels = sorted(df["label"].unique())
    print("\nTraining HMMs for labels:")
    for lbl in unique_labels:
        print(f"  - {lbl}")


    # train one HMM for each label
    for label in unique_labels:
        X = X_all[labels == label]

        if len(X) < N_COMPONENTS * 5:
            print(f"Skipping {label} (not enough samples)")
            continue

        # variance floor
        var = np.var(X, axis=0)
        small_var = var < VAR_FLOOR

        if np.any(small_var):
            noise = np.random.normal(scale=1e-2, size=X.shape)
            X[:, small_var] += noise[:, small_var]

        # HMM training definition
        hmm = GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type=COV_TYPE,
            n_iter=200,
            random_state=RANDOM_STATE,
            verbose=False,
        )

        hmm.fit(X)

        model_path = MODEL_DIR / f"hmm_{label}.pkl"
        joblib.dump(hmm, model_path)

        print(f"Saved HMM for '{label}' → {model_path.name}")

    labels_path = MODEL_DIR / "labels.txt"
    with open(labels_path, "w") as f:
        for lbl in unique_labels:
            f.write(lbl + "\n")

    print("\nTraining complete.")
    print(f"Models saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
