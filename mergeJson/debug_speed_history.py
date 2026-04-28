import json
from pathlib import Path

def main():
    json_path = Path(__file__).parent / "merged_players.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    players = data.get("players", {})
    player_ids = list(players.keys())
    
    if not player_ids:
        print("No players found in JSON.")
        return

    print("Available Player IDs:")
    for pid in player_ids:
        print(f" - {pid}")
    
    selected_id = input("\nEnter Player ID to debug: ").strip().lower()
    
    if selected_id not in players:
        print(f"Error: Player '{selected_id}' not found.")
        return
    
    speed_history = players[selected_id].get("speed_history", [])
    
    if not speed_history:
        print(f"No speed history found for {selected_id}.")
        return

    sorted_speeds = sorted(speed_history)
    
    print(f"\n--- Speed History for {selected_id} (Sorted, Count: {len(sorted_speeds)}) ---")
    for i, speed in enumerate(sorted_speeds):
        # Print in groups or just everything? If it's 66k frames, printing all might be too much.
        # Let's print the distribution or samples if it's too long.
        if len(sorted_speeds) > 100:
            if i < 20 or i > len(sorted_speeds) - 21:
                print(f"{i+1:5}: {speed:.2f} km/h")
            elif i == 20:
                print("  ...")
        else:
            print(f"{i+1:5}: {speed:.2f} km/h")

    # Also show some stats
    if sorted_speeds:
        print(f"\nMin: {sorted_speeds[0]:.2f} km/h")
        print(f"Max: {sorted_speeds[-1]:.2f} km/h")
        print(f"99.5th Percentile: {sorted_speeds[int(len(sorted_speeds)*0.995)]:.2f} km/h")
        print(f"Average: {sum(sorted_speeds)/len(sorted_speeds):.2f} km/h")

        # Locate which video the max speed came from
        print("\n--- Source of Max Speed ---")
        raw_max = sorted_speeds[-1]
        json_replaced_dir = json_path.parent / "json_replaced"
        
        found_clips = []
        if json_replaced_dir.exists():
            # Scan all analysis files to find the one with this max speed
            for json_file in json_replaced_dir.glob("*_analysis.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        clip_data = json.load(f)
                    
                    p_stats = clip_data.get("player_stats", {}).get(selected_id)
                    if p_stats:
                        clip_max = p_stats.get("max_speed_kmh", 0.0)
                        # Use a small epsilon for float comparison
                        if abs(clip_max - raw_max) < 1e-7:
                            video_num = json_file.name.split("_")[0]
                            found_clips.append(video_num)
                except Exception:
                    continue
            
            if found_clips:
                # Sort numerically
                found_clips.sort(key=lambda x: int(x) if x.isdigit() else 0)
                print(f"Max Speed ({raw_max:.2f} km/h) appeared in {len(found_clips)} video(s):")
                for v in found_clips:
                    print(f" - Video {v}")
            else:
                print("Could not trace the max speed to a specific video in 'json_replaced'.")
        else:
            print("Directory 'json_replaced' not found. Cannot trace source.")

if __name__ == "__main__":
    main()
