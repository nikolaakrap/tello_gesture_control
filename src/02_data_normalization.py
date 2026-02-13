import pandas as pd
import numpy as np
from pathlib import Path


# config
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "normalized_samples.csv"


# normalization
def normalize_hand(coords):
    """
    coords: (21, 3)
    - center on wrist (landmark 0)
    - scale by distance wrist -> middle MCP (landmark 9)
    """
    coords = coords.copy()
    wrist = coords[0]

    coords -= wrist

    scale = np.linalg.norm(coords[9])
    if scale < 1e-6:
        scale = 1.0

    coords /= scale
    return coords


def main():
    csv_files = sorted(RAW_DIR.glob("hand_samples_*.csv"))

    if not csv_files:
        raise RuntimeError("No raw CSV files found in data/raw")

    print(f"Found {len(csv_files)} raw CSV files")

    frames = []

    for csv_path in csv_files:
        print(f"Loading {csv_path.name}")
        df = pd.read_csv(csv_path)

        # basic sanity check
        expected_cols = 4 + 21 * 3
        if df.shape[1] != expected_cols:
            print(f"  Skipping {csv_path.name} (wrong column count)")
            continue

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"Total samples: {len(data)}")

    normalized_rows = []

    for _, row in data.iterrows():
        try:
            coords = []
            for i in range(21):
                coords.append([
                    row[f"x{i}"],
                    row[f"y{i}"],
                    row[f"z{i}"],
                ])
            coords = np.array(coords, dtype=np.float32)

            coords = normalize_hand(coords)

            flat = coords.flatten()

            out_row = {
                "sample_id": row["sample_id"],
                "user": row["user"],
                "label": row["label"],
                "timestamp": row["timestamp"],
            }

            for i, v in enumerate(flat):
                out_row[f"f{i}"] = v

            normalized_rows.append(out_row)

        # skip any corrupted rows
        except Exception:
            continue

    out_df = pd.DataFrame(normalized_rows)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved normalized dataset to:")
    print(f"  {OUT_CSV}")
    print(f"Final sample count: {len(out_df)}")


if __name__ == "__main__":
    main()
