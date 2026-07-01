"""
Player selection module - handles interactive player selection UI and logic
"""
import cv2

try:
    import ctypes
except ImportError:
    ctypes = None


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


def _is_ctrl_pressed():
    """Check whether Ctrl is currently held down on Windows."""
    if ctypes is None:
        return False

    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
    except Exception:
        return False


def _get_all_player_ids_in_game(tracks):
    """Collect all unique player IDs across all frames in the entire game."""
    all_ids = set()
    for frame_players in tracks.get('players', []):
        all_ids.update(int(pid) for pid in frame_players.keys())
    return all_ids


def _parse_typed_player_ids(typed_input, all_available_ids=None):
    """Parse typed player IDs from user input (works game-wide like --players).
    
    Supports formats like:
    - "1 2 3"
    - "1,2,3"
    - "1, 2, 3"
    - "1"
    
    Args:
        typed_input: String of player IDs to parse
        all_available_ids: Set of valid player IDs in the entire game (for validation)
    
    Returns a set of player IDs that either:
    - Match IDs from all_available_ids (if provided), or
    - Are any integers (if all_available_ids is None, for --players style behavior)
    """
    if not typed_input.strip():
        return set()
    
    # Replace commas with spaces and split
    tokens = typed_input.replace(',', ' ').split()
    valid_ids = set()
    
    for token in tokens:
        token = token.strip()
        if not token.isdigit():
            print(f"[select-target] Invalid input: '{token}' is not a number, skipping.")
            continue
        player_id = int(token)
        if all_available_ids and player_id not in all_available_ids:
            print(f"[select-target] Player {player_id} not found in any frame of the game.")
            continue
        valid_ids.add(player_id)
    
    return valid_ids


def _first_frame_with_selected_players(tracks, selected_player_ids, max_frame_index=None):
    """Find the earliest frame where all selected player IDs are visible.

    If no shared frame exists before max_frame_index, returns None.
    """
    if not selected_player_ids:
        return None

    selected_set = {int(pid) for pid in selected_player_ids}
    if max_frame_index is None:
        max_frame_index = len(tracks.get('players', [])) - 1

    max_frame_index = min(max_frame_index, len(tracks.get('players', [])) - 1)
    for frame_idx in range(0, max_frame_index + 1):
        frame_players = tracks['players'][frame_idx]
        if selected_set.issubset(set(frame_players.keys())):
            return frame_idx

    return None


def select_target_player_ids(video_frames, tracks, preferred_frame_index=None, return_selected_frame=False):
    """Interactively select multiple players from the entire game (click to toggle select/deselect or type IDs).
    
    Returns a list of selected stable player IDs.
    If return_selected_frame=True, returns (selected_ids, selected_frame_index).
    
    Keyboard controls:
    - Left Click: toggle player selection (only on current frame display)
    - T or I: enter ID input mode (applies to whole game)
    - ENTER: confirm selection (in normal mode) or add typed IDs (in input mode)
    - BACKSPACE: previous frame by 1 (or 5 with Ctrl)
    - ESC or Q: cancel selection
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

    # Collect all player IDs in the entire game for type-based selection
    all_game_player_ids = _get_all_player_ids_in_game(tracks)
    
    state = {
        'selected_player_ids': set(),
        'current_frame': frame_with_players,
        'input_mode': False,
        'typed_input': '',
    }
    window_name = "Select Target Players (Click/Type to select | ENTER=confirm | BACKSPACE=prev | ESC=cancel)"

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

        # Display mode and instructions
        if state['input_mode']:
            instructions_text = "TYPE MODE (applies to whole game): Enter IDs like '1,3,5' or '1 3 5' | ENTER=add | ESC=cancel"
            input_display = f"Typed: {state['typed_input']}"
            instructions_color = (0, 165, 255)  # Orange
        else:
            instructions_text = "Click frame players or press T to type IDs (applies to whole game) | ENTER=confirm | BACKSPACE=prev frame"
            input_display = None
            instructions_color = (255, 255, 255)

        cv2.putText(
            preview,
            f"Frame {state['current_frame']} - {instructions_text}",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            instructions_color,
            2,
        )

        if input_display:
            cv2.putText(
                preview,
                input_display,
                (25, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        if state['selected_player_ids']:
            selected_text = f"Selected for whole game: {sorted(state['selected_player_ids'])}"
            y_offset = 105 if input_display else 70
            cv2.putText(
                preview,
                selected_text,
                (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        # Handle input mode
        if state['input_mode']:
            if key in (13, 10):  # Enter - confirm typed IDs
                if state['typed_input'].strip():
                    # Parse typed IDs (works game-wide)
                    typed_ids = _parse_typed_player_ids(state['typed_input'], all_available_ids=all_game_player_ids)
                    if typed_ids:
                        state['selected_player_ids'].update(typed_ids)
                        print(f"[select-target] Added player IDs from input (game-wide): {sorted(typed_ids)}")
                    state['typed_input'] = ''
                state['input_mode'] = False
            elif key in (27, ord('q')):  # Esc or q - cancel input mode
                state['typed_input'] = ''
                state['input_mode'] = False
            elif key in (8, 127):  # Backspace - delete last character
                state['typed_input'] = state['typed_input'][:-1]
            elif 48 <= key <= 57 or key in (44, 32):  # 0-9, comma, space
                state['typed_input'] += chr(key)
            continue

        # Handle normal mode
        if key in (13, 10):  # Enter
            if state['selected_player_ids']:
                selected_ids = sorted(list(state['selected_player_ids']))
                selected_frame = _first_frame_with_selected_players(
                    tracks,
                    selected_ids,
                    max_frame_index=state['current_frame'],
                )
                if selected_frame is None:
                    selected_frame = state['current_frame']
                cv2.destroyWindow(window_name)
                print(f"[select-target] Selected stable player IDs: {selected_ids}")
                if return_selected_frame:
                    return selected_ids, selected_frame
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
        elif key in (8, 127):  # Backspace / Delete-style back navigation
            step_back = 5 if _is_ctrl_pressed() else 1
            previous_frame = max(0, state['current_frame'] - step_back)
            if previous_frame == state['current_frame']:
                continue

            state['current_frame'] = previous_frame
            frame_players_current = tracks['players'][state['current_frame']]
            if len(frame_players_current) == 0:
                print(f"[select-target] Frame {state['current_frame']} has no players, moving backward again if needed...")
            continue
        elif key == ord('t') or key == ord('i'):  # T or I - enter input mode
            state['input_mode'] = True
            state['typed_input'] = ''
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
