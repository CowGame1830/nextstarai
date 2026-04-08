# Player Attribute Calculation Guide

This project takes raw tracking data from YOLOv8 and converts it into **Football Manager (FM)** style player attributes (10-200 scale).

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

## 📂 File Structure

- **`calculate_unified_attributes.py`**: **Main Entry Point.** Run this to process all player checkpoints and see the final FM-scale attributes.
- **`core/`**:
    - `attribute_cal.py`: The core formulas for attribute calculation.
    - `unify_stats.py`: Logic for merging multiple player IDs/checkpoints into one profile.
- **`tools/`**:
    - `find_max.py`: Scanner to find top speed/acceleration in tracking data.
    - `web_scraping.py`: Tool to fetch real attributes from FMPlayer.net.
- **`input/`**: Directory for your tracking JSON files.
- **`player_data/`**: Directory for player lists and scraped results.

## 🚀 Quick Usage

To see your unified player attributes (merging all data for each player), run:
```bash
python calculate_unified_attributes.py
```
