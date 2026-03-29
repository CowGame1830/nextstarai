import json
import glob
import os
from datetime import datetime

from database import SessionLocal
from models import Player, PlayerStats, Model1Attributes

def get_latest_json(folder_name="output_data", prefix="player_stats"):
    base_dir = os.path.dirname(os.path.dirname(__file__))  # root project
    folder = os.path.join(base_dir, folder_name)

    files = glob.glob(os.path.join(folder, f"{prefix}_*.json"))

    if not files:
        raise FileNotFoundError(f"No {prefix} JSON found in {folder}")

    return max(files, key=os.path.getctime)


def main():
    session = SessionLocal()

    # Find lastest files
    player_stats_file = get_latest_json("output_data", "player_stats")
    print("Using file:", player_stats_file)

    with open(player_stats_file) as f:
        data = json.load(f)

    # ---------------- CREATE PLAYERS FIRST ----------------
    for key, player in data.items():
        if key == "match_summary":
            continue

        player_id = str(player["player_id"])

        existing_player = session.get(Player, player_id)
        if not existing_player:
            new_player = Player(
                id=player_id,
                name=f"Player {player_id}",
                team=player.get("team", ""),
                position=player.get("position", ""),
                age=player.get("age", 0),
                nationality=player.get("nationality", ""),
                height_cm=player.get("height_cm", 0),
                weight_kg=player.get("weight_kg", 0),
                preferred_foot=player.get("preferred_foot", ""),
                photo=player.get("photo", ""),
                create_at=datetime.now(),
                update_at=datetime.now()
            )
            session.add(new_player)

    session.flush()  # flush for player_id to ready in FK

    # ---------------- INSERT PLAYER STATS & MODEL 1 ----------------
    for key, player in data.items():
        if key == "match_summary":
            continue

        player_id = str(player["player_id"])

        # PLAYER STATS
        stats = PlayerStats(
            player_id=player_id,
            season="2026",
            appearances=player.get("total_frames_tracked", 0),
            goals=0,
            assists=0,
            minutes_played=0,
            rating=player.get("avg_speed_kmh", 0),
            key_passes=0,
            create_at=datetime.now(),
            update_at=datetime.now()
        )
        session.add(stats)

        # MODEL 1 ATTRIBUTES
        model1 = Model1Attributes(
            player_id=player_id,
            avg_speed_kmh=player.get("avg_speed_kmh", 0),
            max_speed_kmh=player.get("max_speed_kmh", 0),
            sprint_speed_kmh=player.get("sprint_speed_kmh", 0),
            total_distance_m=player.get("total_distance_m", 0),
            stamina_percentage=player.get("stamina_percentage", 0),
            acceleration=player.get("max_acceleration", 0),
            deceleration=0,  # ยังไม่มี
            agility_score=0,  # ยังไม่มี
            jump_count=player.get("jump_count", 0),
            normalized_attribute={
                "sprint_time": player.get("sprint_time_seconds", 0),
                "sprint_distance": player.get("total_sprint_distance_m", 0),
                "frames": player.get("total_frames_tracked", 0)
            },
            create_at=datetime.now()
        )
        session.add(model1)

    session.commit()
    session.close()
    print("Insert completed!")


if __name__ == "__main__":
    main()