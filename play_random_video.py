#!/usr/bin/env python3
"""
Random Video Player for Football Analysis Output Videos

This script randomly selects and plays a video from the output_videos folder.
Supported formats: .mp4, .avi, .mov, .mkv, .webm

Usage:
    python play_random_video.py           # Play a random video
    python play_random_video.py --repeat  # Continuously play random videos
    python play_random_video.py --list    # List all available videos
"""

import os
import sys
import random
import subprocess
import time
import shutil
from pathlib import Path

import cv2


def get_output_videos_dir():
    """Get the output_videos directory path."""
    current_dir = Path(__file__).parent
    output_videos_dir = current_dir / "output_videos"
    return output_videos_dir


def find_video_files(directory):
    """Find all video files in the directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    videos = []
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return videos
    
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            videos.append(file)
    
    return sorted(videos)


def play_video_and_wait(video_path):
    """Play a video file and wait for it to finish.
    
    Args:
        video_path: Path to the video file
    """
    try:
        print(f"Playing: {video_path.name}")
        play_video_fullscreen_opencv(video_path)
        
        print("Video finished.\n")
    except Exception as e:
        print(f"Error playing video: {e}\n")


def play_video_fullscreen_opencv(video_path):
    """Play a video in a fullscreen OpenCV window and wait until it ends."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    window_name = "Football Video Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps = capture.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / fps)) if fps and fps > 0 else 33

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord('q'), 27):
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    capture.release()
    cv2.destroyWindow(window_name)


def main():
    """Main function."""
    output_videos_dir = get_output_videos_dir()
    videos = find_video_files(output_videos_dir)
    
    if not videos:
        print(f"No video files found in '{output_videos_dir}'")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv")
        return
    
    # Parse command line arguments
    repeat = '--repeat' in sys.argv
    list_only = '--list' in sys.argv
    
    if list_only:
        print(f"\nFound {len(videos)} video(s) in '{output_videos_dir}':\n")
        for i, video in enumerate(videos, 1):
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"  {i}. {video.name} ({size_mb:.1f} MB)")
        return
    
    if repeat:
        print(f"Playing videos randomly from '{output_videos_dir}'")
        print(f"Total videos: {len(videos)}")
        print("Press Ctrl+C to stop.\n")
        
        try:
            while True:
                selected_video = random.choice(videos)
                play_video_and_wait(selected_video)
                print()  # Blank line between videos
        except KeyboardInterrupt:
            print("\n\nStopped.")
    else:
        selected_video = random.choice(videos)
        play_video_and_wait(selected_video)


if __name__ == "__main__":
    main()
