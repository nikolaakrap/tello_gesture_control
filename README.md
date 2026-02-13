# Gesture-Based Real-Time Drone Control using HMM

This project implements a real-time gesture-based control system for a
DJI Tello drone using a camera and Hidden Markov Models (HMM).
The system detects hand landmarks using MediaPipe, classifies gestures
using Gaussian HMMs, and sends control commands to the drone via UDP.

------------------------------------------------------------------------

## Features

-   Real-time hand landmark detection (MediaPipe)
-   Landmark normalization
-   Gesture classification using Gaussian Hidden Markov Models
-   Reference rule-based gesture classifier
-   Real-time DJI Tello drone control
-   Latency measurement and evaluation
-   Confusion matrix visualization and performance metrics

------------------------------------------------------------------------

## Project Structure

    data/
    ├── raw/                # Raw recorded samples
    ├── processed/          # Normalized dataset (CSV)
    ├── models/             # Saved HMM models
    ├── figures/            # Generated plots

    src/
    ├── 01_data_collection.py
    ├── 02_data_normalization.py
    ├── 03_train_hmm.py
    ├── 04_realtime_test.py
    ├── 05_drone_control.py
    ├── 06_model_evaluation.py
    ├── 07_latency_test.py
    ├── 08_latency_analysis.py

    requirements.txt

------------------------------------------------------------------------

## Gesture Mapping

The following gestures are mapped to drone commands:

| Gesture | Drone Action |
|----------|--------------|
| Closed fist | Move forward |
| Open palm | Move backward |
| Index finger up | Move left |
| Pinky up | Move right |
| Horns (index + pinky) | Move up |
| Horns + thumb | Move down |
| Thumbs up | Takeoff |
| Thumbs down | Land |

------------------------------------------------------------------------

## Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/nikolaakrap/tello_gesture_control.git
cd tello_gesture_control
```

------------------------------------------------------------------------

### 2. Create a virtual environment (recommended)

**macOS / Linux:**

```
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```
python -m venv venv
venv\Scripts\activate
```

------------------------------------------------------------------------

### 3. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

## How to Use

### Step 1 -- Data Collection

```
python src/01_data_collection.py
```

### Step 2 -- Data Normalization

```
python src/02_data_normalization.py
```

### Step 3 -- Train HMM Models

```
python src/03_train_hmm.py
```

### Step 4 -- Evaluate Model Performance

```
python src/06_model_evaluation.py
```

### Step 5 -- Real-Time Gesture Testing

```
python src/04_realtime_test.py
```

### Step 6 -- Real-Time Drone Control

```
python src/05_drone_control.py
```

Make sure the drone is powered on and you are connected to its Wi-Fi
network.

### Step 7 -- Latency Measurement

```
python src/07_latency_test.py
```

### Step 8 -- Latency Analysis

```
python src/08_latency_analysis.py
```

------------------------------------------------------------------------

## Evaluation Metrics

-   Accuracy
-   Macro F1-score
-   False Positive Rate (noise class)
-   End-to-end latency

------------------------------------------------------------------------

## Model Details

-   Gaussian Hidden Markov Models (4 hidden states per gesture)
-   One model per gesture (generative classification approach)
-   Sliding window for temporal consistency
-   Fixed random seed for reproducibility

------------------------------------------------------------------------

## Notes

-   The evaluation script performs its own train/test split.
-   Real-time performance depends on camera frame rate and system
    performance.
-   Drone command frequency (\~20 Hz) contributes to total system
    latency.
