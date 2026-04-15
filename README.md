# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Model Benchmark (Model 1 Performance Test)
Use `benchmark_model.py` to test if model 1 works reliably on real match clips (for example, Wolves team clips).

Quick benchmark on a video:

```bash
python benchmark_model.py --model models/clean_label.pt --video input_videos/10secVideo.mp4
```

Quick smoke test (first 200 frames only):

```bash
python benchmark_model.py --model models/clean_label.pt --video input_videos/10secVideo.mp4 --max-frames 200
```

Benchmark with labeled validation set (`data.yaml`) for mAP:

```bash
python benchmark_model.py --model models/clean_label.pt --video input_videos/10secVideo.mp4 --data path/to/data.yaml
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

- `output_videos/<video_name>/`
- `output_data/<video_name>/`

Output files are named with the input video name and timestamp, for example:

- `<video_name>_<timestamp>_output.avi`
- `<video_name>_<timestamp>_analysis.json`
- `<video_name>_player_stats_<timestamp>.json`
- `<video_name>_enhanced_player_stats_<timestamp>.json`

This removes the old checkpoint video naming and makes it easier to match every output back to its source video.