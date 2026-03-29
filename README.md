# Player Attribute Calculation Guide

This project takes raw tracking data from YOLOv8 and converts it into **Football Manager (FM)** style player attributes (1-20 scale).

---

## 🏃 How We Calculate Each Attribute

Since the tracking data can sometimes have "impossible" speeds (like 200+ km/h) due to camera calibration, we use **Relative Scaling**. This means we compare every player in your video to the *best* performer in that specific video.

### 1. Pace (Top Speed)
This measures the player's highest recorded speed.
*   **How it works:** We find the fastest player in your video and give them a **20**. Every other player is scored relative to that top speed.
*   **Formula logic:** `(Your Top Speed / Video's Max Speed) * 20`

### 2. Acceleration (Burst)
This measures how quickly a player can "explode" into a run.
*   **How it works:** We look at the single highest acceleration moment for the player.
*   **Formula logic:** `(Your Max Acceleration / Video's Max Acceleration) * 20`

### 3. Stamina (Endurance)
This measures how much "gas" the player has in the tank and how much they used.
*   **How it works:** It's an average of two things:
    1.  **Work Intensity:** How much distance the player covered per frame (did they move a lot?).
    2.  **Energy Left:** The "stamina_percentage" directly from your tracking data.
*   **Formula logic:** If you moved a lot and still have high energy, you get a higher score.

### 4. Work Rate (Effort)
This measures a player's "hustle"—how often they are actively trying to make things happen.
*   **How it works:** We look at two factors:
    1.  **Standard Movement:** How much distance you covered overall.
    2.  **Sprinting Effort:** How much of your time was spent in a full sprint versus walking.
*   **Formula logic:** A player who runs high-intensity sprints frequently will get a **20**.

---

## 📂 File Breakdown

1.  **`attribute_cal.py`**: The "Brain" of the project. This is where the 4 main functions live.
2.  **`print_attributes.py`**: The "Display" tool. Run this to see a nice table of all your players and their scores.
3.  **`find_max.py`**: The "Scanner". It finds the top speed/acceleration in your specific video so we can scale everyone else correctly.

## 🚀 Quick Usage

To see your player attributes, just run:
```bash
python print_attributes.py
```
