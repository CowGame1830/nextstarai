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


def detect_selected_players_disappeared(
    tracks,
    selected_player_ids,
    start_frame=0,
    max_missing_frames=12,
    min_presence_frames=8,
):
    """Check if selected player IDs truly disappeared (not short occlusion).

    Returns: (disappeared, frame_num_when_disappearance_started)
    """
    if not selected_player_ids:
        return False, None

    selected_set = set(int(pid) for pid in selected_player_ids)
    presence_frames = 0
    missing_streak = 0

    for frame_num in range(start_frame, len(tracks['players'])):
        frame_players = tracks['players'][frame_num]
        frame_player_ids = set(frame_players.keys())
        selected_in_frame = selected_set & frame_player_ids

        if selected_in_frame:
            presence_frames += 1
            missing_streak = 0
            continue

        # Avoid false positives before the selected player has been reliably visible.
        if presence_frames < max(1, int(min_presence_frames)):
            continue

        missing_streak += 1
        if missing_streak >= max(1, int(max_missing_frames)):
            disappearance_start = frame_num - missing_streak + 1
            return True, disappearance_start

    return False, None
