import json
import os
import glob
from core.unify_stats import get_unified_stats
from core.attribute_cal import calculate_pace, calculate_acceleration, calculate_stamina, calculate_work_rate

INPUT_DIR = r"d:\nextstarAI\attribute_calculate\input"


def find_latest_input(directory):
    """Auto-find the most recent merged_checkpoints JSON in the input folder."""
    pattern = os.path.join(directory, "merged_checkpoints_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def calculate_attributes_from_file(json_path):
    """Load a merged_checkpoints JSON and calculate FM-scale attributes for the player."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # All player IDs in the file = same physical person (ID-switching)
    # get_unified_stats merges everything into one single profile dict
    stats = get_unified_stats(data)

    if not stats:
        print("No valid player data found.")
        return

    # --- Calculate FM-scale Attributes ---
    pace  = calculate_pace(stats.get("max_speed_kmh", 0))
    acc   = calculate_acceleration(stats.get("max_acceleration_enhanced", 0))
    wr    = calculate_work_rate(stats.get("avg_speed_kmh", 0), stats.get("sprint_percentage", 0))
    stam  = calculate_stamina(stats.get("stamina_percentage", 0))

    # --- Print Results ---
    print("\n" + "=" * 55)
    print(f"{'PLAYER ATTRIBUTES  (10–200 scale)':^55}")
    print(f"  Source: {os.path.basename(json_path)}")
    print("=" * 55)
    print(f"{'Attribute':<20} | {'Value':>6}")
    print("-" * 35)
    print(f"{'Pace':<20} | {pace:>6}")
    print(f"{'Acceleration':<20} | {acc:>6}")
    print(f"{'Work Rate':<20} | {wr:>6}")
    print(f"{'Stamina':<20} | {stam:>6}")
    print("=" * 55)

    print("\n--- Raw Stats Used ---")
    print(f"  Max Speed       : {stats.get('max_speed_kmh', 0):.2f} km/h")
    print(f"  Max Accel       : {stats.get('max_acceleration_enhanced', 0):.2f} m/s²")
    print(f"  Avg Speed       : {stats.get('avg_speed_kmh', 0):.2f} km/h")
    print(f"  Sprint %        : {stats.get('sprint_percentage', 0) * 100:.1f}%")
    print(f"  Stamina         : {stats.get('stamina_percentage', 0):.1f}%")
    print(f"  Total Distance  : {stats.get('total_distance_m', 0):.1f} m")
    print(f"  Total Frames    : {stats.get('total_frames_tracked', 0)}")
    print()


def main():
    json_path = find_latest_input(INPUT_DIR)

    if not json_path or not os.path.exists(json_path):
        print(f"Error: No merged_checkpoints JSON found in {INPUT_DIR}")
        return

    print(f"Using: {json_path}")
    calculate_attributes_from_file(json_path)


if __name__ == "__main__":
    main()
