# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Installation & Setup

Follow these steps to set up the project locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/CowGame1830/nextstarai.git
   cd nextstarai
   ```

2. **Set up a Virtual Environment (Optional but recommended):**
   * **Windows:**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*(Note: Recommended Python version is 3.13.x. The requirements file is configured to install PyTorch CPU version to ensure fast inference setup on CPU. If you have a compatible NVIDIA GPU, you may want to install PyTorch with CUDA support instead.)*

## Model Benchmark (Model 1 Performance Test)
Use `benchmark_model.py` to test if model 1 works reliably on real match clips (for example, Wolves team clips).

Quick benchmark on a video:

```bash
python benchmark_model.py --model models/clean_label.pt --video input_videos/8.mp4
```

Quick smoke test (first 200 frames only):

```bash
python benchmark_model.py --model models/clean_label.pt --video input_videos/8.mp4 --max-frames 200
```

Benchmark with labeled validation set (`data.yaml`) for mAP:

```bash
python benchmark_model.py --model models/clean_label.pt --video input_videos/8.mp4 --data path/to/data.yaml
```

The script saves a JSON report in `output_data/` with:
- Inference FPS and milliseconds per frame
- Total detections and detections per frame
- Per-class detection confidence and frame presence ratio
- Optional mAP50/mAP50-95 when a validation set is provided

## Select One Target Player (Interactive)
You can run the full pipeline but analyze/export only one selected player.

Interactive mode (click player bbox, then press Enter):

```bash
python main.py --select-target
```

Interactive mode on a specific frame index:

```bash
python main.py --select-target --target-frame 120
```

Manual mode without click (known stable id):

```bash
python main.py --players 10
```

Notes:
- `--select-target` uses stabilized player IDs, so the selected player is tracked consistently across the whole video.
- If OpenCV GUI is not available in your environment, use `--players` with a known ID.

## Runtime Optimization Options
Run the main pipeline with custom input/model paths:

```bash
python main.py --video input_videos/5.mp4 --model models/clean_label.pt
```

You can combine with target-selection mode:

```bash
python main.py --video input_videos/5.mp4 --model models/clean_label.pt --select-target
```

Performance notes:
- Tracker now uses stronger matching settings to reduce ID switches during occlusion/crossing.
- Output JSON writing will automatically use `orjson` (if installed) for faster export.

## Batch Processing Multiple Videos
You can process many videos in one run with `--videos`.

Example:

```bash
python main.py --videos input_videos/1.mp4 input_videos/2.mp4 input_videos/3.mp4
```

If you have a long list, you can pass 30 paths the same way. Each video is processed independently and writes its own outputs under:

- `output_videos/`
- `output_data/`

Output files are named with the input video name and timestamp, for example:

- `<video_name>_<timestamp>_output.avi`
- `<video_name>_<timestamp>_analysis.json`
- `<video_name>_player_stats_<timestamp>.json`
- `<video_name>_enhanced_player_stats_<timestamp>.json`

This removes the old checkpoint video naming and makes it easier to match every output back to its source video.