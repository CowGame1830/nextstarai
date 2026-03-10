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
from pose_estimator.advanced_jump_detector_clean import AdvancedJumpDetector
from minimap_generator import MinimapGenerator
import json
from datetime import datetime
import argparse


def parse_selected_player_ids(players_args):
    """Parse optional player ids from CLI args.

    Supports both formats:
    - --players 1 2 33
    - --players 1,2,33
    """
    if not players_args:
        return None

    parsed_ids = set()
    for raw_value in players_args:
        for token in str(raw_value).split(','):
            token = token.strip()
            if not token:
                continue
            if not token.isdigit():
                raise ValueError(f"Invalid player id: '{token}'. Use integers only.")
            parsed_ids.add(int(token))

    return sorted(parsed_ids) if parsed_ids else None


def filter_combined_stats_by_players(combined_stats, selected_player_ids=None):
    """Filter combined stats payload to selected players only."""
    if not selected_player_ids:
        return combined_stats

    selected_set = set(int(pid) for pid in selected_player_ids)
    filtered_ids = sorted([pid for pid in combined_stats.get('detected_player_ids', []) if int(pid) in selected_set])

    filtered_player_stats = {}
    original_player_stats = combined_stats.get('player_stats', {})
    for pid in filtered_ids:
        key = f"player_{int(pid)}"
        if key in original_player_stats:
            filtered_player_stats[key] = original_player_stats[key]

    if 'match_summary' in original_player_stats:
        summary = dict(original_player_stats['match_summary'])
        summary['total_players'] = len(filtered_ids)
        filtered_player_stats['match_summary'] = summary

    filtered_enhanced_stats = {}
    original_enhanced_stats = combined_stats.get('enhanced_stats', {})
    for pid in filtered_ids:
        key = f"player_{int(pid)}"
        if key in original_enhanced_stats:
            filtered_enhanced_stats[key] = original_enhanced_stats[key]

    filtered_advanced_jump_stats = {}
    original_advanced_jump_stats = combined_stats.get('advanced_jump_stats', {})
    for pid in filtered_ids:
        key = str(int(pid))
        if key in original_advanced_jump_stats:
            filtered_advanced_jump_stats[key] = original_advanced_jump_stats[key]

    return {
        "timestamp": combined_stats.get("timestamp"),
        "input_video": combined_stats.get("input_video"),
        "total_detected_players": len(filtered_ids),
        "detected_player_ids": filtered_ids,
        "player_stats": filtered_player_stats,
        "enhanced_stats": filtered_enhanced_stats,
        "advanced_jump_stats": filtered_advanced_jump_stats,
    }


def filter_tracks_for_selected_players(tracks, selected_player_ids=None):
    """Return a visualization-friendly tracks dict filtered to selected players only."""
    if not selected_player_ids:
        return tracks

    selected_set = set(int(pid) for pid in selected_player_ids)
    filtered_tracks = {}

    for object_name, object_tracks in tracks.items():
        if object_name == 'players':
            filtered_player_frames = []
            for frame_players in object_tracks:
                filtered_frame = {
                    player_id: track_info
                    for player_id, track_info in frame_players.items()
                    if int(player_id) in selected_set
                }
                filtered_player_frames.append(filtered_frame)
            filtered_tracks[object_name] = filtered_player_frames
        else:
            # Keep non-player objects unchanged (ball/referees) for pipeline compatibility.
            filtered_tracks[object_name] = object_tracks

    return filtered_tracks


def main(enable_minimap=False, verbose_debug=False, export_player_ids=None):
    # Read Video
    video_frames = read_video('input_videos/10secVideo.mp4')
    print(f"จำนวนเฟรมใน video_frames = {len(video_frames)}")

    # Initialize Tracker
    tracker = Tracker('models/clean_label.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')
    tracks = tracker.stabilize_player_ids(tracks)
    print("Player ID stabilization completed")
    print(f"[DEBUG] จำนวนเฟรมใน tracks['players'] = {len(tracks['players'])}")
    print(f"[DEBUG] จำนวนเฟรมใน tracks['ball'] = {len(tracks['ball'])}")
    print(f"[DEBUG] จำนวนเฟรมใน tracks['referees'] = {len(tracks['referees'])}")
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator with advanced jump detection
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.set_video_frames(video_frames)  # Set frames for jump detection
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # Initialize advanced jump detector for enhanced jump tracking
    jump_detector = AdvancedJumpDetector()
    print("Advanced jump detection initialized")

    # Initialize Player Statistics Tracker
    player_stats_tracker = PlayerStatsTracker(frame_rate=24)
    player_stats_tracker.update_player_stats(tracks)
    
    # Initialize Minimap Generator only when needed
    minimap_generator = None
    if enable_minimap:
        minimap_generator = MinimapGenerator(minimap_width=350, minimap_height=230)
        print("Minimap generator initialized")

    # Assign Player Teams
    team_assigner = TeamAssigner()
    
    # Find first frame with players for team color assignment
    first_frame_with_players = None
    for frame_idx, player_track in enumerate(tracks['players']):
        if len(player_track) > 0:
            first_frame_with_players = frame_idx
            break
    
    if first_frame_with_players is not None:
        try:
            team_assigner.assign_team_color(video_frames[first_frame_with_players], 
                                          tracks['players'][first_frame_with_players])
            print(f"Team colors assigned using frame {first_frame_with_players}")
        except Exception as e:
            print(f"Warning: Could not assign team colors: {e}")
            # Set default team colors
            team_assigner.team_colors = {1: (255, 0, 0), 2: (0, 0, 255)}
    else:
        print("Warning: No players found in any frame for team assignment")
        team_assigner.team_colors = {1: (255, 0, 0), 2: (0, 0, 255)}
    
    # Set minimap team colors to match assigned team colors
    if enable_minimap and minimap_generator and hasattr(team_assigner, 'team_colors') and team_assigner.team_colors:
        # Convert BGR to RGB for minimap (OpenCV uses BGR, minimap uses RGB)
        team1_color = team_assigner.team_colors.get(1, (255, 0, 0))  # Default red
        team2_color = team_assigner.team_colors.get(2, (0, 0, 255))  # Default blue
        # Convert BGR to RGB
        team1_rgb = (team1_color[2], team1_color[1], team1_color[0])
        team2_rgb = (team2_color[2], team2_color[1], team2_color[0])
        minimap_generator.set_team_colors(team1_rgb, team2_rgb)
        print(f"Minimap team colors set: Team 1: {team1_rgb}, Team 2: {team2_rgb}")
    
    # Assign teams to all players
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            try:
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                     track['bbox'],
                                                     player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            except Exception as e:
                # Assign default team if assignment fails
                tracks['players'][frame_num][player_id]['team'] = 1
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(1, (255, 0, 0))

    
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        current_ball = tracks['ball'][frame_num].get(1, {}) if frame_num < len(tracks['ball']) else {}
        ball_bbox = current_ball.get('bbox')

        if ball_bbox is None:
            # No ball in this frame: keep previous team control if available
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)
            continue

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # Use previous team control or default to team 1 if no previous data
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)  # Default to team 1
    
    team_ball_control = np.array(team_ball_control)

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
    output_video_frames = tracker.draw_annotations(video_frames, visual_tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # Process advanced jump detection for all frames FIRST
    print("Processing advanced jump detection...")
    
    for frame_num in range(len(tracks['players'])):
        if frame_num < len(video_frames):
            # Get player tracks for current frame
            current_player_tracks = {}
            for player_id, track_info in tracks['players'][frame_num].items():
                current_player_tracks[player_id] = {'bbox': track_info['bbox']}
            
            # Process jump detection
            _, jump_data = jump_detector.process_frame(video_frames[frame_num], current_player_tracks, frame_num)
            
            # Update tracks with advanced jump data
            for player_id, jump_info in jump_data.items():
                if player_id in tracks['players'][frame_num]:
                    tracks['players'][frame_num][player_id]['advanced_jump'] = jump_info.get('is_jumping', False)
                    tracks['players'][frame_num][player_id]['jump_phase'] = jump_info.get('jump_phase', 'unknown')
                    tracks['players'][frame_num][player_id]['jump_height_advanced'] = jump_info.get('jump_height', 0)
                    tracks['players'][frame_num][player_id]['jump_confidence'] = jump_info.get('jump_confidence', 0)

    ## Draw Speed and Distance (now with updated jump data)
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, visual_tracks)
    
    # Optional minimap overlay (disabled by default)
    if enable_minimap and minimap_generator:
        print("Adding minimap overlay to video frames...")
        output_video_frames = minimap_generator.draw_minimap_with_stats(
            output_video_frames,
            visual_tracks,
            position='bottom_center',
            team_ball_control=team_ball_control
        )
        print("Minimap overlay completed - integrated into main video")
    
    # Save one combined JSON file for all analysis outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_player_ids = sorted({pid for frame_players in tracks['players'] for pid in frame_players.keys()})

    combined_stats = {
        "timestamp": timestamp,
        "input_video": "input_videos/10secVideo.mp4",
        "total_detected_players": len(all_player_ids),
        "detected_player_ids": [int(pid) for pid in all_player_ids],
        "player_stats": player_stats_tracker.build_stats_payload(timestamp=timestamp),
        "enhanced_stats": speed_and_distance_estimator.build_enhanced_stats_data(all_player_ids=all_player_ids),
        "advanced_jump_stats": jump_detector.get_player_jump_stats()
    }

    combined_stats = filter_combined_stats_by_players(combined_stats, export_player_ids)

    if export_player_ids:
        print(f"Exporting selected players only: {combined_stats['detected_player_ids']}")
    else:
        print("Exporting stats for all detected players")

    os.makedirs("output_data", exist_ok=True)
    combined_stats_file = os.path.join("output_data", f"match_stats_{timestamp}.json")
    with open(combined_stats_file, "w", encoding="utf-8") as f:
        json.dump(combined_stats, f, indent=2, ensure_ascii=False)
    print(f"Combined statistics saved to: {combined_stats_file}")

    # Save video with all features including minimap (same filename as before)
    save_video(output_video_frames, 'output_videos/output_video.avi')

    # Print advanced jump detection summary
    jump_stats = jump_detector.get_player_jump_stats()
    print("\n=== Advanced Jump Detection Summary ===")
    for player_id, stats in jump_stats.items():
        print(f"Player {player_id}: {stats['total_jumps']} jumps detected, Max height: {stats['max_jump_height']:.1f}px")
    print("==========================================\n")


###############################################################################

# MediaPipe Skeleton Visualization - Output JPG images
def main_skeleton_jpg():
    """
    Process video with MediaPipe pose estimation showing only skeleton/bones
    and output individual frames as JPG images
    """
    from pose_estimator.mediapipe_jump_detector import MediaPipeJumpDetector

    # Read Video
    video_frames = read_video('input_videos/10secVideo.mp4')
    print(f"Total frames: {len(video_frames)}")

    # Initialize Tracker
    tracker = Tracker('models/clean_label.pt')

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
    
    print(f"\n✓ All frames saved to: {output_dir}/")
    print(f"✓ Total images: {len(video_frames)}")
    print("✓ Skeleton visualization only - no UI elements")


###############################################################################


#test 1 frame
def main3():
    # Read Video
    # video_frames = read_video('input_videos/CHEvLIV1.mp4')
    video_frames = read_video('input_videos/1.mp4')

    # Initialize Tracker
    tracker = Tracker('models/model_2_0.pt')

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
        "--minimap",
        action="store_true",
        help="Enable minimap overlay"
    )
    parser.add_argument(
        "--verbose-debug",
        action="store_true",
        help="Enable verbose debug output"
    )

    args = parser.parse_args()
    selected_player_ids = parse_selected_player_ids(args.players)

    # Default behavior: export all players when --players is omitted.
    main(
        enable_minimap=args.minimap,
        verbose_debug=args.verbose_debug,
        export_player_ids=selected_player_ids
    )