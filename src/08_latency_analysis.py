import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# config
CSV_PATH = Path("data/latency_trials.csv")
OUTPUT_DIR = Path("data/figures")


# load data
df = pd.read_csv(CSV_PATH)

# remove empty gesture rows
df = df[df["gesture"].notna()]
df = df[df["gesture"] != ""]

# remove any invalid latency rows
df = df[df["end_to_end_latency_ms"].notna()]
df = df[df["end_to_end_latency_ms"] > 0]

print("Valid trials:", len(df))

if len(df) == 0:
    raise RuntimeError("No valid latency data found.")


# basic stats
print("\n=== OVERALL LATENCY STATISTICS ===")

mean_latency = df["end_to_end_latency_ms"].mean()
std_latency = df["end_to_end_latency_ms"].std()
min_latency = df["end_to_end_latency_ms"].min()
max_latency = df["end_to_end_latency_ms"].max()
median_latency = df["end_to_end_latency_ms"].median()

print(f"Mean:    {mean_latency:.2f} ms")
print(f"Std:     {std_latency:.2f} ms")
print(f"Median:  {median_latency:.2f} ms")
print(f"Min:     {min_latency:.2f} ms")
print(f"Max:     {max_latency:.2f} ms")


# pre-gesture stats
print("\n=== PER-GESTURE LATENCY ===")

grouped = df.groupby("gesture")["end_to_end_latency_ms"]

gesture_stats = grouped.agg(["count", "mean", "std", "min", "max", "median"])
print(gesture_stats)

gesture_stats.to_csv(OUTPUT_DIR / "latency_stats_per_gesture.csv")


# plots
plt.figure(figsize=(10,6))
df.boxplot(column="end_to_end_latency_ms", by="gesture", grid=False)
plt.title("Latency per Gesture")
plt.suptitle("")
plt.ylabel("End-to-End Latency (ms)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "latency_boxplot.png")
plt.close()


plt.figure(figsize=(8,5))
plt.hist(df["end_to_end_latency_ms"], bins=15)
plt.xlabel("End-to-End Latency (ms)")
plt.ylabel("Frequency")
plt.title("Latency Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "latency_histogram.png")
plt.close()

print("\nPlots saved to:", OUTPUT_DIR)
