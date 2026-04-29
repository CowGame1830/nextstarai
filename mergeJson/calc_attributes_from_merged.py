"""
calc_attributes_from_merged.py
===============================
Reads merged_players.json (output of merge_all_clips.py) and calculates
FM-scale attributes for each player using the same formulas as attribute_cal.py.

Output: merged_players_attributes.json

Usage:
    python calc_attributes_from_merged.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add parent to path so we can import core/attribute_cal.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent   # attribute_calculate/
sys.path.insert(0, str(ROOT))

from core.attribute_cal import (
    calculate_pace,
    calculate_acceleration,
    calculate_stamina,
    calculate_work_rate,
)

# ---------------------------------------------------------------------------
# Input / Output paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
INPUT_FILE  = BASE_DIR / "merged_players.json"
OUTPUT_FILE = BASE_DIR / "merged_players_attributes.json"

FPS = 30  # assumed frame rate for sprint percentage


# ---------------------------------------------------------------------------
# Derive stats needed by attribute_cal from a merged player block
# ---------------------------------------------------------------------------

def get_percentile(arr, p):
    if not arr: return 0
    sorted_arr = sorted(arr)
    idx = int(len(sorted_arr) * (p / 100))
    return sorted_arr[min(idx, len(sorted_arr) - 1)]


def derive_stats(player_data: dict) -> dict:
    """
    Convert the merged player block into the stat dict expected by
    calculate_pace / calculate_acceleration / calculate_work_rate / calculate_stamina.
    """
    frames = player_data.get("total_frames_tracked", 0)
    total_seconds = frames / FPS if frames > 0 else 0

    sprint_time = player_data.get("sprint_time_seconds", 0.0)
    sprint_pct = min(1.0, sprint_time / total_seconds) if total_seconds > 0 else 0.0

    speed_history = player_data.get("speed_history", [])
    raw_max = player_data.get("max_speed_kmh", 0.0)
    
    # -----------------------------------------------------------------------
    # Filter max speed using speed_history (99.5th percentile)
    # -----------------------------------------------------------------------
    if speed_history and len(speed_history) >= 10:
        clean_max = get_percentile(speed_history, 99.5)
    else:
        clean_max = raw_max

    # -----------------------------------------------------------------------
    # Calculate Decay Rate (Fatigue Resistance)
    # Compare peak performance in first 25% vs last 25% of history
    # -----------------------------------------------------------------------
    decay_rate = 1.0
    if len(speed_history) >= 100:
        n = len(speed_history)
        q1 = speed_history[:n//4]
        q4 = speed_history[-(n//4):]
        peak_start = get_percentile(q1, 90)
        peak_end   = get_percentile(q4, 90)
        if peak_start > 5.0: # Only calculate if they actually moved in Q1
            decay_rate = peak_end / peak_start

    # -----------------------------------------------------------------------
    # Calculate Work Rate Signals (Action Density)
    # Based on Observed Time (frames where history is available)
    # -----------------------------------------------------------------------
    jogging_percentage = 0.0
    burst_frequency = 0.0
    
    if len(speed_history) > 0:
        # Use total_seconds calculated from total_frames_tracked to avoid errors with downsampled history
        duration_minutes = total_seconds / 60.0
        
        # 1. Jogging % (> 7 km/h) - Continuity
        jog_frames = len([s for s in speed_history if s > 7.0])
        jogging_percentage = jog_frames / len(speed_history)
        
        # 2. HIR % (> 19.8 km/h)
        hir_frames = len([s for s in speed_history if s > 19.8])
        hir_percentage = hir_frames / len(speed_history)
        
        # 3. Burst Frequency (Starts per minute)
        # We detect "significant actions" where speed crosses a high intensity threshold
        bursts = 0
        in_burst = False
        for s in speed_history:
            if s >= 19.8: # High Intensity Running threshold (Bradley et al.)
                if not in_burst:
                    bursts += 1
                    in_burst = True
            elif s < 15.0: # Must drop below 15 to reset the burst
                in_burst = False
        
        burst_frequency = bursts / duration_minutes if duration_minutes > 0 else 0.0

    return {
        "max_speed_kmh":          clean_max,
        "max_acceleration_enhanced": player_data.get("max_acceleration", 0.0),
        "avg_speed_kmh":          player_data.get("avg_speed_kmh", 0.0),
        "sprint_percentage":      sprint_pct,
        "hir_percentage":         hir_percentage if len(speed_history) > 0 else 0.0,
        "stamina_percentage":     player_data.get("stamina_percentage", 0.0),
        "decay_rate":             decay_rate,
        "jogging_percentage":     jogging_percentage,
        "burst_frequency":        burst_frequency,
        "total_distance_m":       player_data.get("total_distance_m", 0.0),
        "total_frames_tracked":   frames,
        "sprint_time_seconds":    sprint_time,
        "jump_count":             player_data.get("jump_count", 0),
        "clips_appeared":         player_data.get("clips_appeared", []),
    }



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        print(f"[ERROR] merged_players.json not found at {INPUT_FILE}")
        print("  Run merge_all_clips.py first.")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    players: dict = data.get("players", {})
    total_clips: int = data.get("total_clips_merged", 0)

    print(f"Loaded {len(players)} player(s) from {total_clips} merged clips.\n")

    results = {}

    sep = "=" * 60
    for player_key, player_data in players.items():
        stats = derive_stats(player_data)

        pace = calculate_pace(stats["max_speed_kmh"])
        acc  = calculate_acceleration(stats["max_acceleration_enhanced"])
        wr   = calculate_work_rate(stats["avg_speed_kmh"], stats["jogging_percentage"], stats["burst_frequency"])
        stam = calculate_stamina(stats["sprint_percentage"], stats["hir_percentage"], stats["decay_rate"])

        clips_count = len(stats["clips_appeared"])

        print(sep)
        print(f"  {player_key.upper():^56}")
        print(f"  Appeared in {clips_count} clip(s)")
        print(sep)
        print(f"  {'Attribute':<22} | {'Score':>6}")
        print(f"  {'-'*22}-+-{'-'*6}")
        print(f"  {'Pace':<22} | {pace:>6}")
        print(f"  {'Acceleration':<22} | {acc:>6}")
        print(f"  {'Work Rate':<22} | {wr:>6}")
        print(f"  {'Stamina':<22} | {stam:>6}")
        print()
        print(f"  --- Raw Stats ---")
        print(f"  Max Speed        : {stats['max_speed_kmh']:.2f} km/h")
        print(f"  Max Acceleration : {stats['max_acceleration_enhanced']:.3f} m/s2")
        print(f"  Avg Speed        : {stats['avg_speed_kmh']:.2f} km/h")
        print(f"  Sprint %         : {stats['sprint_percentage'] * 100:.1f}%")
        print(f"  Stamina          : {stats['stamina_percentage']:.1f}%")
        print(f"  Total Distance   : {stats['total_distance_m']:.1f} m")
        print(f"  Jump Count       : {stats['jump_count']}")
        print(f"  Frames Tracked   : {stats['total_frames_tracked']}")
        print()

        results[player_key] = {
            "attributes": {
                "pace":         pace,
                "acceleration": acc,
                "work_rate":    wr,
                "stamina":      stam,
            },
            "raw_stats": {
                "max_speed_kmh":       stats["max_speed_kmh"],
                "max_acceleration":    stats["max_acceleration_enhanced"],
                "avg_speed_kmh":       stats["avg_speed_kmh"],
                "sprint_percentage":   round(stats["sprint_percentage"] * 100, 2),
                "stamina_percentage":  stats["stamina_percentage"],
                "total_distance_m":    stats["total_distance_m"],
                "sprint_time_seconds": stats["sprint_time_seconds"],
                "jump_count":          stats["jump_count"],
                "total_frames_tracked": stats["total_frames_tracked"],
                "clips_appeared":      stats["clips_appeared"],
                "clips_count":         clips_count,
            },
        }

    output = {
        "total_clips_source": total_clips,
        "total_players":      len(results),
        "players":            results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(sep)
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
