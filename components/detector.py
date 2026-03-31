"""
Detection module - handles jump detection, ID switching, and player disappearance detection
"""


def apply_advanced_jump_detection(video_frames, tracks, jump_detector):
    """Apply advanced jump detection to all frames."""
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
