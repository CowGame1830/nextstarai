"""
Assignment module - handles team color assignment, team assignment, and ball control assignment
"""


def assign_team_colors_from_first_player_frame(team_assigner, video_frames, tracks):
    """Assign team colors based on players in the first frame."""
    first_frame_with_players = None
    for frame_idx, frame_players in enumerate(tracks.get('players', [])):
        if len(frame_players) > 0:
            first_frame_with_players = frame_idx
            break
    
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
    """Assign teams to all players across all frames."""
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            try:
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            except Exception:
                team = 1

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (255, 0, 0))


def assign_ball_control(tracks):
    """Assign ball control to players based on proximity."""
    import numpy as np
    from player_ball_assigner import PlayerBallAssigner
    
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
