# Cricket-biomechanics

# Cricket Biomechanics Analysis Pipeline

## 1. Project Overview
This project is a computer vision pipeline designed to analyze cricket bowling/batting technique from a side-on video. It uses pose estimation to track key body joints and calculate performance-related metrics in real-time, focusing on interpretable movement data rather than just visual overlays.

* **Input:** Side-on video of a cricket player.
* **Output:** Annotated video with skeleton overlay and JSON data containing frame-wise biomechanical metrics.

### Video Selection Rationale
* **Suitability:** The selected video was chosen because it provides a clear side-on view (approx. 90 degrees to the pitch), which is essential for accurate 2D angle calculation of the knee and elbow. It simulates a real-world use case of a user recording via a mobile phone.
* **Challenges:** Like many phone-recorded videos, it presents minor challenges such as motion blur on the fast-moving arm and occasional self-occlusion during the delivery stride, which tests the model's robustness.

---

## 2. Approach & Model Selection
* **Model Used:** MediaPipe Pose (Google).
* **Why this model?**
    * **Efficiency:** It is lightweight and optimized for CPU inference, making it accessible for deployment on mobile or edge devices (the core product foundation).
    * [cite_start]**Robustness:** It offers a 33-point skeletal topology that captures feet and hand positioning accurately enough for general technique analysis without the heavy computational cost of OpenPose[cite: 63].

---

## 3. Metrics Defined
I implemented three specific metrics to evaluate player technique. [cite_start]These were chosen because they directly correlate with performance and injury prevention[cite: 33, 41].

### A. Front Knee Angle (Stability)
* **Definition:** The angle formed by the **Hip**, **Knee**, and **Ankle** of the front landing leg.
* **Why it matters:** In fast bowling, a "braced front leg" (straight knee) at the moment of release allows for efficient energy transfer from the run-up to the ball. A collapsing knee (angle decreasing) dissipates energy and reduces bowling speed.

### B. Elbow Extension Angle (Legality & Technique)
* **Definition:** The angle formed by the **Shoulder**, **Elbow**, and **Wrist**.
* **Why it matters:**
    * *Bowling:* Critical for monitoring "chucking." A bowler cannot extend their arm by more than 15 degrees during the delivery swing.
    * *Batting:* Indicates a full extension of the arms during shots (e.g., a cover drive), ensuring maximum power.

### C. Torso Lean (Balance)
* **Definition:** The angle of the torso (mid-shoulder to mid-hip line) relative to the vertical axis.
* **Why it matters:** Proper body lean ensures balance. Leaning too far back during delivery can cause lower back injuries (stress fractures), while leaning too far forward before release can shorten the delivery stride.

---

## 4. Observations & Limitations
[cite_start]During the development and testing of this pipeline, I observed the following[cite: 46]:

* **Jitter:** The raw keypoints exhibit high-frequency jitter, especially in the extremities (wrist/ankle) during fast motion. This results in noisy metric graphs.
* **Occlusion:** In a side-on view, the back arm or leg is frequently occluded by the torso. The model sometimes "hallucinates" these points or loses tracking temporarily.
* **Motion Blur:** Fast bowling arms often appear as a blur in standard 30fps video, reducing keypoint accuracy at the most critical moment (ball release).

---

## 5. Model Improvement Thinking
*(As requested in Step 4 of the assignment)*

If I had more time and resources, here is how I would improve the pipeline to production quality:

### How to Improve Accuracy & Adaptability
1.  **Temporal Smoothing:** I would implement a **One-Euro Filter** or **Savitzky-Golay Filter** on the raw $(x, y)$ coordinates before calculating angles. [cite_start]This would significantly reduce the jitter observed in the output[cite: 50].
2.  **3D Lifting:** Instead of relying on 2D projection angles (which distort based on camera angle), I would use a 2D-to-3D lifting model (like VideoPose3D) to estimate true joint angles.

### [cite_start]Data Collection Strategy [cite: 52]
* **High Frame Rate:** Collect data at 120fps or 240fps (slow motion) to eliminate motion blur on the bowling arm.
* **Diversity:** Capture videos with varying lighting conditions (indoor nets vs. outdoor sun) and background clutter to ensure the model isn't overfitting to clean backgrounds.

### [cite_start]Data Splitting Strategy [cite: 53-57]
To ensure the model generalizes to *new* users and doesn't just memorize the movement of specific players, I would split the data by **Player ID**, not by random frames:
* **Train (70%):** Videos of Players A-M.
* **Validation (15%):** Videos of Players N-Q (used for hyperparameter tuning).
* **Test (15%):** Videos of Players R-Z (completely unseen players to test real-world performance).

### [cite_start]Evaluation Metrics [cite: 58]
I would evaluate the improvement using:
* **PCK (Percentage of Correct Keypoints):** To measure how often the predicted keypoint is within a certain threshold of the ground truth.
* **MPJPE (Mean Per Joint Position Error):** To measure the average Euclidean distance between predicted and ground truth joints.

---

## 6. How to Run
1.  Install dependencies: `pip install opencv-python mediapipe numpy`
2.  Place your video in the folder and rename it to `input_video.mp4` (or update `VIDEO_PATH` in `main.py`).
3.  Run the script: `python main.py`
4.  Output will be saved as `output_overlay.mp4` and `keypoints.json`.
