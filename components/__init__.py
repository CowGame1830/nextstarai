"""
Components package - modular football analysis components
"""

from .video_manager import get_available_videos, select_videos_interactive
from .player_selector import (
    select_target_player_ids,
    select_target_player_id,
    parse_selected_player_ids,
    _first_frame_with_players,
    _point_in_bbox
)
from .assigner import (
    assign_team_colors_from_first_player_frame,
    assign_teams_to_all_tracks,
    assign_ball_control
)
from .detector import (
    detect_id_switch,
    detect_selected_players_disappeared
)
from .stats_processor import (
    dump_json_file,
    filter_combined_stats_by_players,
    filter_tracks_for_selected_players,
    save_checkpoint_data,
    merge_checkpoint_stats
)

__all__ = [
    # video_manager
    'get_available_videos',
    'select_videos_interactive',
    # player_selector
    'select_target_player_ids',
    'select_target_player_id',
    'parse_selected_player_ids',
    '_first_frame_with_players',
    '_point_in_bbox',
    # assigner
    'assign_team_colors_from_first_player_frame',
    'assign_teams_to_all_tracks',
    'assign_ball_control',
    # detector
    'detect_id_switch',
    'detect_selected_players_disappeared',
    # stats_processor
    'dump_json_file',
    'filter_combined_stats_by_players',
    'filter_tracks_for_selected_players',
    'save_checkpoint_data',
    'merge_checkpoint_stats',
]
