"""
Player selection module - handles interactive player selection UI and logic
"""
import cv2


def _point_in_bbox(x, y, bbox):
    """Check if point (x, y) is inside bounding box"""
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _first_frame_with_players(tracks):
    """Find the first frame that contains players"""
    for frame_idx, frame_players in enumerate(tracks.get('players', [])):
        if len(frame_players) > 0:
            return frame_idx
    return None


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

    state = {'selected_player_ids': set(), 'current_frame': frame_with_players}
    window_name = "Select Target Players (Left Click=toggle | ENTER=confirm | NO SELECT+ENTER=skip 5 frames | ESC=cancel)"

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        frame_players_current = tracks['players'][state['current_frame']]
        hit_candidates = []
        for player_id, track_info in frame_players_current.items():
            bbox = track_info.get('bbox')
            if bbox is None:
                continue
            if _point_in_bbox(x, y, bbox):
                x1, y1, x2, y2 = bbox
                area = max(1.0, (x2 - x1) * (y2 - y1))
                hit_candidates.append((area, int(player_id)))

        if not hit_candidates:
            return

        hit_candidates.sort(key=lambda item: item[0])
        clicked_player_id = hit_candidates[0][1]
        
        # Toggle: if already selected, remove; otherwise add
        if clicked_player_id in state['selected_player_ids']:
            state['selected_player_ids'].discard(clicked_player_id)
        else:
            state['selected_player_ids'].add(clicked_player_id)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)
    except cv2.error as e:
        print(f"[select-target] OpenCV UI unavailable: {e}")
        return None

    while True:
        frame_players_current = tracks['players'][state['current_frame']]
        preview = video_frames[state['current_frame']].copy()

        for player_id, track_info in frame_players_current.items():
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
            f"Frame {state['current_frame']} - (Click=toggle) | ENTER=confirm | NO SELECT+ENTER=skip 5 | ESC=cancel",
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
            else:
                # No selection - skip 5 frames forward
                state['current_frame'] += 5
                if state['current_frame'] >= len(tracks['players']):
                    print("[select-target] Reached end of video without selecting players.")
                    cv2.destroyWindow(window_name)
                    return None
                frame_players_current = tracks['players'][state['current_frame']]
                if len(frame_players_current) == 0:
                    print(f"[select-target] Frame {state['current_frame']} has no players, skipping...")
                    continue
        elif key in (27, ord('q')):  # Esc or q
            cv2.destroyWindow(window_name)
            print("[select-target] Selection canceled by user.")
            return None


def select_target_player_id(video_frames, tracks, preferred_frame_index=None):
    """Deprecated: Use select_target_player_ids() instead.
    
    This function now calls select_target_player_ids() and returns the first selected player ID.
    """
    result = select_target_player_ids(video_frames, tracks, preferred_frame_index)
    if result is None or len(result) == 0:
        return None
    return result[0]
