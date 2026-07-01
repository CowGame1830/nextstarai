"""
Stats processor module - handles checkpoint creation, filtering, and merging
"""
import json
from datetime import datetime

try:
    import orjson
except ImportError:
    orjson = None


def dump_json_file(path, data):
    """Save data to JSON file (using orjson if available for better performance)."""
    if orjson is not None:
        with open(path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def filter_combined_stats_by_players(combined_stats, selected_player_ids=None):
    """Filter combined stats payload to selected players only."""
    if not selected_player_ids:
        selected_player_ids = combined_stats.get('detected_player_ids', [])

    selected_set = set(int(pid) for pid in selected_player_ids)
    filtered_ids = sorted([pid for pid in combined_stats.get('detected_player_ids', []) if int(pid) in selected_set])

    merged_player_stats = {}
    original_player_stats = combined_stats.get('player_stats', {})
    original_enhanced_stats = combined_stats.get('enhanced_stats', {})

    duplicate_enhanced_keys = {
        'player_id',
        'max_speed_kmh',
        'avg_speed_kmh',
        'max_acceleration',
        'total_distance_m',
        'jump_count',
        'stamina_diff',
        'final_stamina_percentage',
        'jumps_detected',
    }

    for pid in filtered_ids:
        key = f"player_{int(pid)}"
        merged_stats = dict(original_player_stats.get(key, {}))
        enhanced_stats = dict(original_enhanced_stats.get(key, {}))

        # The player key already identifies the player, so keep the payload lean.
        merged_stats.pop('player_id', None)
        merged_stats.pop('stamina_diff', None)

        # Keep the richer time-series fields, but avoid repeating the same
        # metrics in both sections of the export.
        for duplicate_key in duplicate_enhanced_keys:
            enhanced_stats.pop(duplicate_key, None)

        merged_stats.update(enhanced_stats)
        if merged_stats:
            merged_player_stats[key] = merged_stats

    if 'match_summary' in original_player_stats:
        summary = dict(original_player_stats['match_summary'])
        summary.pop('timestamp', None)
        summary['total_players'] = len(filtered_ids)
        merged_player_stats['match_summary'] = summary

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
        "player_stats": merged_player_stats,
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


def save_checkpoint_data(combined_stats, selected_player_ids, checkpoint_name="checkpoint_1"):
    """Save current stats as checkpoint to memory."""
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
