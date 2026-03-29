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

try:
    import orjson
except ImportError:
    orjson = None


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


def _point_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _first_frame_with_players(tracks):
    for frame_idx, frame_players in enumerate(tracks.get('players', [])):
        if len(frame_players) > 0:
            return frame_idx
    return None


def select_target_player_ids(video_frames, tracks, preferred_frame_index=None):
    """Interactively select multiple players from a frame (click to toggle select/deselect).
    
    Returns a list of selected stable player IDs.
    """
    frame_with_players = preferred_frame_index
    if frame_with_players is None:
        frame_with_players = _first_frame_with_players(tracks)

    if frame_with_players is None:
        print("[select-target] No frame contains players. Cannot select target.")
        return None

    if frame_with_players < 0 or frame_with_players >= len(video_frames):
        print(f"[select-target] Invalid frame index: {frame_with_players}")
        return None

    frame_players = tracks['players'][frame_with_players]
    if len(frame_players) == 0:
        print(f"[select-target] Frame {frame_with_players} has no players.")
        return None

    state = {'selected_player_ids': set()}
    window_name = "Select Target Players (Left Click to toggle, ENTER confirm, ESC cancel)"

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        hit_candidates = []
        for player_id, track_info in frame_players.items():
            bbox = track_info.get('bbox')
            if bbox is None:
                continue
            if _point_in_bbox(x, y, bbox):
                x1, y1, x2, y2 = bbox
                area = max(1.0, (x2 - x1) * (y2 - y1))
                hit_candidates.append((area, int(player_id)))

        if not hit_candidates:
            print("[select-target] Click inside a player bounding box.")
            return

        hit_candidates.sort(key=lambda item: item[0])
        clicked_player_id = hit_candidates[0][1]
        
        # Toggle: if already selected, remove; otherwise add
        if clicked_player_id in state['selected_player_ids']:
            state['selected_player_ids'].discard(clicked_player_id)
            print(f"[select-target] Deselected player ID: {clicked_player_id}")
        else:
            state['selected_player_ids'].add(clicked_player_id)
            print(f"[select-target] Selected player ID: {clicked_player_id}")

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
    except cv2.error as e:
        print(f"[select-target] OpenCV UI unavailable: {e}")
        return None

    while True:
        preview = video_frames[frame_with_players].copy()

        for player_id, track_info in frame_players.items():
            bbox = track_info.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = map(int, bbox)

            is_selected = int(player_id) in state['selected_player_ids']
            color = (0, 255, 0) if is_selected else (255, 255, 0)
            thickness = 3 if is_selected else 2

            cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(preview, f"ID {int(player_id)}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.putText(
            preview,
            f"Frame {frame_with_players} - (Click=toggle) | ENTER=confirm | ESC=cancel",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if state['selected_player_ids']:
            selected_text = f"Selected: {sorted(state['selected_player_ids'])}"
            cv2.putText(
                preview,
                selected_text,
                (25, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if state['selected_player_ids']:
                selected_ids = sorted(list(state['selected_player_ids']))
                cv2.destroyWindow(window_name)
                print(f"[select-target] Selected stable player IDs: {selected_ids}")
                return selected_ids
            print("[select-target] Please select at least one player before pressing Enter.")
        elif key in (27, ord('q')):  # Esc or q
            cv2.destroyWindow(window_name)
            print("[select-target] Selection canceled by user.")
            return None


# Keep backward-compatible alias for single player selection
def select_target_player_id(video_frames, tracks, preferred_frame_index=None):
    """Deprecated: Use select_target_player_ids() instead.
    
    This function now calls select_target_player_ids() and returns the first selected player ID.
    """
    result = select_target_player_ids(video_frames, tracks, preferred_frame_index)
    if result is None or len(result) == 0:
        return None
    return result[0]


def dump_json_file(path, data):
    if orjson is not None:
        with open(path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def assign_team_colors_from_first_player_frame(team_assigner, video_frames, tracks):
    first_frame_with_players = _first_frame_with_players(tracks)
    if first_frame_with_players is None:
        print("Warning: No players found in any frame for team assignment")
        team_assigner.team_colors = {1: (255, 0, 0), 2: (0, 0, 255)}
        return

    try:
        team_assigner.assign_team_color(
            video_frames[first_frame_with_players],
            tracks['players'][first_frame_with_players],
        )
        print(f"Team colors assigned using frame {first_frame_with_players}")
    except Exception as e:
        print(f"Warning: Could not assign team colors: {e}")
        team_assigner.team_colors = {1: (255, 0, 0), 2: (0, 0, 255)}


def assign_teams_to_all_tracks(team_assigner, video_frames, tracks):
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            try:
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            except Exception:
                team = 1

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (255, 0, 0))


def assign_ball_control(tracks):
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        current_ball = tracks['ball'][frame_num].get(1, {}) if frame_num < len(tracks['ball']) else {}
        ball_bbox = current_ball.get('bbox')

        if ball_bbox is None:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
            continue

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

    return np.array(team_ball_control)


def apply_advanced_jump_detection(video_frames, tracks, jump_detector):
    print("Processing advanced jump detection...")
    for frame_num, frame_players in enumerate(tracks['players']):
        if frame_num >= len(video_frames):
            continue

        current_player_tracks = {
            player_id: {'bbox': track_info['bbox']}
            for player_id, track_info in frame_players.items()
        }

        _, jump_data = jump_detector.process_frame(video_frames[frame_num], current_player_tracks, frame_num)

        for player_id, jump_info in jump_data.items():
            if player_id not in frame_players:
                continue

            frame_players[player_id]['advanced_jump'] = jump_info.get('is_jumping', False)
            frame_players[player_id]['jump_phase'] = jump_info.get('jump_phase', 'unknown')
            frame_players[player_id]['jump_height_advanced'] = jump_info.get('jump_height', 0)
            frame_players[player_id]['jump_confidence'] = jump_info.get('jump_confidence', 0)


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


def detect_id_switch(tracks, min_continuous_frames=30):
    """Detect if ID switch occurred by analyzing player ID continuity.
    
    Returns a tuple: (id_switch_detected, problematic_frame, missing_player_ids)
    """
    player_ids_by_frame = []
    for frame_players in tracks['players']:
        player_ids_by_frame.append(set(frame_players.keys()))
    
    # Find frames where many players suddenly disappear
    for frame_num in range(1, len(player_ids_by_frame)):
        prev_ids = player_ids_by_frame[frame_num - 1]
        curr_ids = player_ids_by_frame[frame_num]
        
        if len(prev_ids) == 0:
            continue
            
        disappeared = prev_ids - curr_ids
        disappear_ratio = len(disappeared) / len(prev_ids)
        
        # If >30% of players suddenly disappear, likely ID switch
        if disappear_ratio > 0.3 and len(disappeared) >= 2:
            return True, frame_num, disappeared
    
    return False, None, set()


def detect_selected_players_disappeared(tracks, selected_player_ids, start_frame=0):
    """Check if selected player IDs disappear during video processing.
    
    Returns: (disappeared, frame_num_when_disappeared)
    """
    if not selected_player_ids:
        return False, None
    
    selected_set = set(int(pid) for pid in selected_player_ids)
    
    for frame_num in range(start_frame, len(tracks['players'])):
        frame_players = tracks['players'][frame_num]
        frame_player_ids = set(frame_players.keys())
        
        # Check if any selected players exist in this frame
        selected_in_frame = selected_set & frame_player_ids
        
        # If none of the selected players are in frame, it's a disappearance
        if len(selected_in_frame) == 0 and frame_num > start_frame:
            return True, frame_num
    
    return False, None


def save_checkpoint_data(combined_stats, selected_player_ids, checkpoint_name="checkpoint_1"):
    """Save current stats as checkpoint to memory file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_data = {
        "checkpoint_name": checkpoint_name,
        "timestamp": timestamp,
        "selected_player_ids": selected_player_ids,
        "stats": combined_stats
    }
    return checkpoint_data


def merge_checkpoint_stats(checkpoint_data_list, timestamp):
    """Merge multiple checkpoint data into one combined output."""
    merged_stats = {
        "timestamp": timestamp,
        "input_video": checkpoint_data_list[0]['stats'].get("input_video"),
        "total_checkpoints": len(checkpoint_data_list),
        "checkpoints": []
    }
    
    all_player_ids = set()
    
    for i, checkpoint in enumerate(checkpoint_data_list):
        cp_info = {
            "checkpoint_name": checkpoint['checkpoint_name'],
            "timestamp": checkpoint['timestamp'],
            "selected_player_ids": checkpoint['selected_player_ids'],
            "player_stats": checkpoint['stats'].get('player_stats', {}),
            "enhanced_stats": checkpoint['stats'].get('enhanced_stats', {}),
            "advanced_jump_stats": checkpoint['stats'].get('advanced_jump_stats', {})
        }
        merged_stats["checkpoints"].append(cp_info)
        all_player_ids.update(checkpoint['selected_player_ids'] or [])
    
    merged_stats["all_selected_player_ids"] = sorted([int(pid) for pid in all_player_ids])
    return merged_stats


def main(
    enable_minimap=False,
    verbose_debug=False,
    export_player_ids=None,
    select_target=False,
    target_frame_index=None,
    input_video_path='input_videos/5.mp4',
    model_path='models/clean_label.pt',
    allow_id_switch_reselect=True,
):
    # Read Video
    video_frames = read_video(input_video_path)
    print(f"จำนวนเฟรมใน video_frames = {len(video_frames)}")

    # Initialize Tracker
    tracker = Tracker(model_path)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')
    tracks = tracker.stabilize_player_ids(tracks)


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
    assign_team_colors_from_first_player_frame(team_assigner, video_frames, tracks)
    
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
    output_video_frames = tracker.draw_annotations(video_frames, visual_tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # Process advanced jump detection for all frames FIRST
    apply_advanced_jump_detection(video_frames, tracks, jump_detector)

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
        "input_video": input_video_path,
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
            reselected_ids = select_target_player_ids(
                video_frames=video_frames,
                tracks=tracks,
                preferred_frame_index=disappeared_frame,
            )
            
            last_disappear_frame = disappeared_frame
            
            # Keep looping through re-selections if players keep disappearing
            while reselected_ids is not None and len(reselected_ids) > 0:
                checkpoint_count += 1
                print(f"\nRe-selected new players (Checkpoint {checkpoint_count}): {reselected_ids}")
                
                # Filter tracks and regenerate stats for new selection
                visual_tracks_cp = filter_tracks_for_selected_players(tracks, reselected_ids)
                
                # Regenerate output frames for this checkpoint
                output_video_frames_cp = tracker.draw_annotations(video_frames, visual_tracks_cp, team_ball_control)
                output_video_frames_cp = camera_movement_estimator.draw_camera_movement(output_video_frames_cp, camera_movement_per_frame)
                output_video_frames_cp = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames_cp, visual_tracks_cp)
                
                if enable_minimap and minimap_generator:
                    output_video_frames_cp = minimap_generator.draw_minimap_with_stats(
                        output_video_frames_cp,
                        visual_tracks_cp,
                        position='bottom_center',
                        team_ball_control=team_ball_control
                    )
                
                # Generate checkpoint stats
                all_player_ids_cp = sorted({pid for frame_players in tracks['players'] for pid in frame_players.keys()})
                combined_stats_cp = {
                    "timestamp": timestamp,
                    "input_video": input_video_path,
                    "total_detected_players": len(all_player_ids_cp),
                    "detected_player_ids": [int(pid) for pid in all_player_ids_cp],
                    "player_stats": player_stats_tracker.build_stats_payload(timestamp=timestamp),
                    "enhanced_stats": speed_and_distance_estimator.build_enhanced_stats_data(all_player_ids=all_player_ids_cp),
                    "advanced_jump_stats": jump_detector.get_player_jump_stats()
                }
                combined_stats_cp = filter_combined_stats_by_players(combined_stats_cp, reselected_ids)
                checkpoint_cp_data = save_checkpoint_data(combined_stats_cp, reselected_ids, f"checkpoint_{checkpoint_count}")
                checkpoint_list.append(checkpoint_cp_data)
                
                print(f"Checkpoint {checkpoint_count} saved with players: {combined_stats_cp['detected_player_ids']}")
                
                # Save video for this checkpoint
                output_video_path = f'output_videos/output_video_checkpoint{checkpoint_count}.avi'
                save_video(output_video_frames_cp, output_video_path)
                print(f"Video saved to {output_video_path}")
                
                # Check if these new selected players also disappeared - if yes, auto trigger again
                selected_disappeared_again, disappeared_frame_again = detect_selected_players_disappeared(
                    tracks, reselected_ids, last_disappear_frame
                )
                
                if selected_disappeared_again:
                    print(f"\n  New selected players disappeared at frame {disappeared_frame_again}")
                    print(" AUTO RE-SELECTION: Popping up player selection UI again...\n")
                    last_disappear_frame = disappeared_frame_again
                    
                    # Auto-trigger next re-selection
                    reselected_ids = select_target_player_ids(
                        video_frames=video_frames,
                        tracks=tracks,
                        preferred_frame_index=disappeared_frame_again,
                    )
                else:
                    # New selection is stable - break loop
                    print("\nCurrent selection is stable - no further disappearances detected")
                    reselected_ids = None

    os.makedirs("output_data", exist_ok=True)
    
    # Save merged checkpoint data if multiple checkpoints exist
    if len(checkpoint_list) > 1:
        merged_data = merge_checkpoint_stats(checkpoint_list, timestamp)
        merged_file = os.path.join("output_data", f"merged_checkpoints_{timestamp}.json")
        dump_json_file(merged_file, merged_data)
        print(f"\nMerged checkpoint data saved to: {merged_file}")
    
    # Save original checkpoint files
    combined_stats_file = os.path.join("output_data", f"match_stats_{timestamp}.json")
    dump_json_file(combined_stats_file, combined_stats)
    print(f"First checkpoint statistics saved to: {combined_stats_file}")

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
    video_frames = read_video('input_videos/5.mp4')
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
        "--video",
        default="input_videos/5.mp4",
        help="Input video path"
    )
    parser.add_argument(
        "--model",
        default="models/clean_label.pt",
        help="Model path"
    )
    parser.add_argument(
        "--no-id-switch-reselect",
        action="store_true",
        help="Disable ID switch re-selection feature (off by default)"
    )

    args = parser.parse_args()
    selected_player_ids = parse_selected_player_ids(args.players)

    # Default behavior: export all players when --players is omitted.
    main(
        enable_minimap=args.minimap,
        verbose_debug=args.verbose_debug,
        export_player_ids=selected_player_ids,
        select_target=args.select_target,
        target_frame_index=args.target_frame,
        input_video_path=args.video,
        model_path=args.model,
        allow_id_switch_reselect=not args.no_id_switch_reselect,
    )