from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import os
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from player_stats import PlayerStatsTracker
import json
from datetime import datetime
import argparse
import threading
import time

# Import components
from components import (
    get_available_videos,
    select_videos_interactive,
    select_target_player_ids,
    parse_selected_player_ids,
    assign_team_colors_from_first_player_frame,
    assign_teams_to_all_tracks,
    assign_ball_control,
    detect_id_switch,
    detect_selected_players_disappeared,
    dump_json_file,
    filter_combined_stats_by_players,
    filter_tracks_for_selected_players,
    save_checkpoint_data,
    merge_checkpoint_stats,
    _first_frame_with_players,
)
# Utility functions are now in the components package
# See: components/video_manager.py, components/player_selector.py, components/assigner.py,
#      components/detector.py, components/stats_processor.py


class TerminalLoader:
    """Lightweight terminal spinner for long-running steps."""

    def __init__(self, message, interval=0.12):
        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self.start_time = None

    def _spin(self):
        frames = ["|", "/", "-", "\\"]
        i = 0
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            print(f"\r{frames[i % len(frames)]} {self.message}... {elapsed:5.1f}s", end="", flush=True)
            i += 1
            time.sleep(self.interval)

    def __enter__(self):
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        elapsed = time.time() - self.start_time if self.start_time else 0
        status = "done" if exc_type is None else "failed"
        print(f"\r[{'OK' if exc_type is None else '!!'}] {self.message} ({status} in {elapsed:.1f}s)")


def run_with_loader(message, func, *args, **kwargs):
    with TerminalLoader(message):
        return func(*args, **kwargs)


def main(
    verbose_debug=False,
    export_player_ids=None,
    select_target=False,
    target_frame_index=None,
    input_video_path='input_videos/5.mp4',
    model_path='models/clean_label.pt',
    detection_conf=0.35,
    allow_id_switch_reselect=True,
):
    # Read Video
    video_frames = run_with_loader("Reading video", read_video, input_video_path)
    print(f"จำนวนเฟรมใน video_frames = {len(video_frames)}")

    # Initialize Tracker
    tracker = Tracker(model_path, detection_conf=detection_conf)

    tracks = run_with_loader(
        "Tracking objects",
        tracker.get_object_tracks,
        video_frames,
        read_from_stub=False,
        stub_path='stubs/track_stubs.pkl'
    )
    tracks = run_with_loader("Stabilizing player IDs", tracker.stabilize_player_ids, tracks)


    if select_target:
        selected_player_ids = select_target_player_ids(
            video_frames=video_frames,
            tracks=tracks,
            preferred_frame_index=target_frame_index,
        )
        if selected_player_ids is None or len(selected_player_ids) == 0:
            raise RuntimeError("select-target mode requires selecting at least one player. Use --players to skip manual selection.")
        export_player_ids = selected_player_ids

    print("Player ID stabilization completed")
    print(f"[DEBUG] จำนวนเฟรมใน tracks['players'] = {len(tracks['players'])}")
    print(f"[DEBUG] จำนวนเฟรมใน tracks['ball'] = {len(tracks['ball'])}")
    print(f"[DEBUG] จำนวนเฟรมใน tracks['referees'] = {len(tracks['referees'])}")
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = run_with_loader(
        "Estimating camera movement",
        camera_movement_estimator.get_camera_movement,
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator with advanced jump detection
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.set_video_frames(video_frames)  # Set frames for jump detection
    run_with_loader("Computing speed and distance", speed_and_distance_estimator.add_speed_and_distance_to_tracks, tracks)

    # Initialize Player Statistics Tracker
    player_stats_tracker = PlayerStatsTracker(frame_rate=24)
    player_stats_tracker.update_player_stats(tracks)
    for frame_players in tracks['players']:
        for player_id, track_info in frame_players.items():
            player_stats = player_stats_tracker.player_stats.get(player_id)
            if player_stats is not None:
                track_info['jump_count'] = int(player_stats.get('jump_count', 0))
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    assign_team_colors_from_first_player_frame(team_assigner, video_frames, tracks)
    
    # Assign teams to all players
    assign_teams_to_all_tracks(team_assigner, video_frames, tracks)

    # Assign Ball Acquisition
    team_ball_control = assign_ball_control(tracks)

    if verbose_debug:
        # Debug: list player IDs และ bbox ทุก frame
        print("\n[DEBUG] ตรวจสอบ player IDs และ bbox ทุก frame")
        for frame_num, player_track in enumerate(tracks['players']):
            print(f"\nFrame {frame_num} - จำนวนผู้เล่น: {len(player_track)}")
            for pid, pdata in player_track.items():
                bbox = pdata['bbox']
                team = pdata.get('team', None)
                has_ball = pdata.get('has_ball', False)
                print(f"  PlayerID {pid}: bbox={bbox}, team={team}, has_ball={has_ball}")

        all_player_ids = set()

        for frame_num, player_track in enumerate(tracks['players']):
            all_player_ids.update(player_track.keys())

        print("\n[DEBUG] Player IDs ทั้งหมดที่ detect ได้ตลอดคลิป:")
        print(all_player_ids)
        print(f"จำนวน Player IDs ทั้งหมด = {len(all_player_ids)}")

    # Draw output 
    visual_tracks = filter_tracks_for_selected_players(tracks, export_player_ids)

    if export_player_ids:
        print(f"Visualizing selected players only: {sorted([int(pid) for pid in export_player_ids])}")
    else:
        print("Visualizing all detected players")

    ## Draw object Tracks
    output_video_frames = run_with_loader(
        "Drawing object annotations",
        tracker.draw_annotations,
        video_frames,
        visual_tracks,
        team_ball_control
    )

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    output_video_frames = run_with_loader(
        "Rendering speed and distance overlays",
        speed_and_distance_estimator.draw_speed_and_distance,
        output_video_frames,
        visual_tracks
    )
    
    # Save one combined JSON file for all analysis outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_player_ids = sorted(
        set(speed_and_distance_estimator.all_detected_player_ids)
        | {pid for frame_players in tracks['players'] for pid in frame_players.keys()}
    )

    speed_and_distance_estimator.player_jump_counts = {
        player_id: int(stats.get('jump_count', 0))
        for player_id, stats in player_stats_tracker.player_stats.items()
    }

    def build_combined_stats_payload():
        return {
            "timestamp": timestamp,
            "input_video": input_video_path,
            "total_detected_players": len(all_player_ids),
            "detected_player_ids": [int(pid) for pid in all_player_ids],
            "player_stats": player_stats_tracker.build_stats_payload(timestamp=timestamp),
            "enhanced_stats": speed_and_distance_estimator.build_enhanced_stats_data(all_player_ids=all_player_ids),
            "advanced_jump_stats": {},
        }

    combined_stats = filter_combined_stats_by_players(build_combined_stats_payload(), export_player_ids)

    if export_player_ids:
        print(f"Exporting selected players only: {combined_stats['detected_player_ids']}")
    else:
        print("Exporting stats for all detected players")

    # CHECKPOINT 1: Save first analysis results
    checkpoint_1_data = save_checkpoint_data(combined_stats, export_player_ids, "checkpoint_1")
    checkpoint_list = [checkpoint_1_data]
    checkpoint_count = 1
    
    # Check if selected players disappeared - auto trigger re-selection (no asking)
    if export_player_ids and allow_id_switch_reselect:
        selected_disappeared, disappeared_frame = detect_selected_players_disappeared(tracks, export_player_ids)
        
        if selected_disappeared:
            print("\n" + "="*70)
            print(f"AUTO RE-SELECTION: Selected players disappeared at frame {disappeared_frame}")
            print(f"Selected players: {export_player_ids}")
            print("="*70)
            print(f"First checkpoint saved with players: {combined_stats['detected_player_ids']}")
            print("Automatically popping up player selection UI...\n")
            
            # Auto-trigger re-selection loop - keep asking until player confirms or no disappearance
            reselected_result = select_target_player_ids(
                video_frames=video_frames,
                tracks=tracks,
                preferred_frame_index=disappeared_frame,
                return_selected_frame=True,
            )
            if reselected_result is None:
                reselected_ids = None
                reselected_from_frame = None
            else:
                reselected_ids, reselected_from_frame = reselected_result
            
            # Keep looping through re-selections if players keep disappearing
            while reselected_ids is not None and len(reselected_ids) > 0:
                checkpoint_count += 1
                print(f"\nRe-selected new players (Checkpoint {checkpoint_count}): {reselected_ids} (from frame {reselected_from_frame})")
                
                # Filter tracks and regenerate stats for new selection
                visual_tracks_cp = filter_tracks_for_selected_players(tracks, reselected_ids)
                
                # Regenerate output frames for this checkpoint
                output_video_frames_cp = run_with_loader(
                    f"Checkpoint {checkpoint_count}: drawing annotations",
                    tracker.draw_annotations,
                    video_frames,
                    visual_tracks_cp,
                    team_ball_control
                )
                output_video_frames_cp = camera_movement_estimator.draw_camera_movement(output_video_frames_cp, camera_movement_per_frame)
                output_video_frames_cp = run_with_loader(
                    f"Checkpoint {checkpoint_count}: rendering overlays",
                    speed_and_distance_estimator.draw_speed_and_distance,
                    output_video_frames_cp,
                    visual_tracks_cp
                )
                
                # Generate checkpoint stats
                combined_stats_cp = filter_combined_stats_by_players(build_combined_stats_payload(), reselected_ids)
                checkpoint_cp_data = save_checkpoint_data(combined_stats_cp, reselected_ids, f"checkpoint_{checkpoint_count}")
                checkpoint_list.append(checkpoint_cp_data)
                
                print(f"Checkpoint {checkpoint_count} saved with players: {combined_stats_cp['detected_player_ids']}")
                
                # Save video for this checkpoint
                output_video_path = f'output_videos/output_video_checkpoint{checkpoint_count}.avi'
                run_with_loader(f"Checkpoint {checkpoint_count}: saving video", save_video, output_video_frames_cp, output_video_path)
                print(f"Video saved to {output_video_path}")
                
                # Check if these new selected players also disappeared - if yes, auto trigger again
                selected_disappeared_again, disappeared_frame_again = detect_selected_players_disappeared(
                    tracks, reselected_ids, reselected_from_frame
                )
                
                if selected_disappeared_again:
                    print(f"\n  New selected players disappeared at frame {disappeared_frame_again}")
                    print(" AUTO RE-SELECTION: Popping up player selection UI again...\n")
                    
                    # Auto-trigger next re-selection
                    reselected_result = select_target_player_ids(
                        video_frames=video_frames,
                        tracks=tracks,
                        preferred_frame_index=disappeared_frame_again,
                        return_selected_frame=True,
                    )
                    if reselected_result is None:
                        reselected_ids = None
                        reselected_from_frame = None
                    else:
                        reselected_ids, reselected_from_frame = reselected_result
                else:
                    # New selection is stable - break loop
                    print("\nCurrent selection is stable - no further disappearances detected")
                    reselected_ids = None

    os.makedirs("output_data", exist_ok=True)
    
    # Save merged checkpoint data only
    merged_data = merge_checkpoint_stats(checkpoint_list, timestamp)
    merged_file = os.path.join("output_data", f"merged_checkpoints_{timestamp}.json")
    dump_json_file(merged_file, merged_data)
    print(f"\nMerged checkpoint data saved to: {merged_file}")

    # Save final video output
    run_with_loader("Saving output video", save_video, output_video_frames, 'output_videos/output_video.avi')


###############################################################################

# MediaPipe Skeleton Visualization - Output JPG images
def main_skeleton_jpg():
    """
    Process video with MediaPipe pose estimation showing only skeleton/bones
    and output individual frames as JPG images
    """
    from pose_estimator.mediapipe_jump_detector import MediaPipeJumpDetector

    # Read Video
    video_frames = read_video('input_videos/5.mp4')
    print(f"Total frames: {len(video_frames)}")

    # Initialize Tracker
    tracker = Tracker('models/clean_label.pt', detection_conf=0.35)  # Improved confidence threshold

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Initialize MediaPipe jump detector with skeleton only (no UI)
    jump_detector = MediaPipeJumpDetector(show_skeleton=True, show_ui=False)
    print("MediaPipe initialized - skeleton visualization only")

    # Create output directory for JPG images
    output_dir = 'output_skeleton_jpg'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each frame
    print("Processing frames with skeleton visualization...")
    
    for frame_num in range(len(tracks['players'])):
        if frame_num < len(video_frames):
            # Get player tracks for current frame
            current_player_tracks = {}
            for player_id, track_info in tracks['players'][frame_num].items():
                current_player_tracks[player_id] = {'bbox': track_info['bbox']}
            
            # Process frame with MediaPipe pose detection
            skeleton_frame, jump_data = jump_detector.process_frame(
                video_frames[frame_num], 
                current_player_tracks, 
                frame_num
            )
            
            # Save frame as JPG
            output_path = os.path.join(output_dir, f'skeleton_frame_{frame_num:04d}.jpg')
            cv2.imwrite(output_path, skeleton_frame)
            
            if frame_num % 10 == 0:  # Print progress every 10 frames
                print(f"Processed frame {frame_num}/{len(video_frames)}")
    
    print(f"\nAll frames saved to: {output_dir}/")
    print(f"Total images: {len(video_frames)}")
    print("Skeleton visualization only - no UI elements")


###############################################################################


#test 1 frame
def main3():
    # Read Video
    # video_frames = read_video('input_videos/CHEvLIV5.mp4')
    video_frames = read_video('input_videos/5.mp4')

    # Initialize Tracker
    tracker = Tracker('models/model_2_0.pt', detection_conf=0.35)  # Improved confidence threshold

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # แสดง bbox เฟรมแรก
    first_frame_players = tracks['players'][0]  # เฟรมแรก
    print("=== BBOX ของผู้เล่นในเฟรมแรก ===")
    for player_id, player_info in first_frame_players.items():
        print(f"Player {player_id}: {player_info['bbox']}")

    # วาด bbox ลงบนภาพ
    frame = video_frames[0].copy()
    for player_id, player_info in first_frame_players.items():
        bbox = player_info['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, str(player_id), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # เซฟภาพ
    output_path = "output_test/frame1080_bbox.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Saved frame with bbox to {output_path}")
    print("Frame shape:", frame.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Football analysis pipeline")
    parser.add_argument(
        "--players",
        nargs="*",
        help="Optional player IDs to export as JSON only for selected players. Examples: --players 1 33 or --players 1,33"
    )
    parser.add_argument(
        "--verbose-debug",
        action="store_true",
        help="Enable verbose debug output"
    )
    parser.add_argument(
        "--select-target",
        action="store_true",
        help="Interactively select one target player (click bounding box) and analyze only that player"
    )
    parser.add_argument(
        "--target-frame",
        type=int,
        default=None,
        help="Optional frame index for target selection. Defaults to first frame that contains players"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Input video paths (space-separated). Example: --videos video1.mp4 video2.mp4"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Input video path (single video). Use --videos for multiple videos"
    )
    parser.add_argument(
        "--select-videos",
        action="store_true",
        help="Interactively select videos from input_videos/ directory"
    )
    parser.add_argument(
        "--model",
        default="models/clean_label.pt",
        help="Model path"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detections (0.0-1.0). Higher = fewer false positives but may miss objects. Default: 0.35"
    )
    parser.add_argument(
        "--no-id-switch-reselect",
        action="store_true",
        help="Disable ID switch re-selection feature (off by default)"
    )

    args = parser.parse_args()
    selected_player_ids = parse_selected_player_ids(args.players)

    # Determine which videos to process
    videos_to_process = []
    
    if args.select_videos:
        # Interactive selection of videos
        videos_to_process = select_videos_interactive()
    elif args.videos:
        # Multiple videos from command line
        videos_to_process = args.videos
    elif args.video:
        # Single video from command line
        videos_to_process = [args.video]
    else:
        # Default: interactive selection if no video specified
        print("\nNo video specified. Starting interactive video selection...")
        videos_to_process = select_videos_interactive()
    
    if not videos_to_process:
        print("No videos to process. Exiting.")
        exit(1)
    
    # Process each selected video
    for video_path in videos_to_process:
        print(f"\n{'='*70}")
        print(f"Processing: {video_path}")
        print(f"{'='*70}\n")
        
        try:
            main(
                verbose_debug=args.verbose_debug,
                export_player_ids=selected_player_ids,
                select_target=args.select_target,
                target_frame_index=args.target_frame,
                input_video_path=video_path,
                model_path=args.model,
                detection_conf=args.conf,
                allow_id_switch_reselect=not args.no_id_switch_reselect,
            )
            print(f"Successfully processed: {video_path}\n")
        except Exception as e:
            print(f"Error processing {video_path}: {e}\n")
            continue
    
    print(f"\n{'='*70}")
    print("All videos processed!")
    print(f"{'='*70}")