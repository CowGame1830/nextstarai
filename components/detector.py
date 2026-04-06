"""
Detection module - handles ID switching and player disappearance detection
"""


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
