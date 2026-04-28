"""
merge_all_clips.py
==================
Reads all JSON files from ./json_replaced/ and merges them into a single
output file: ./merged_players.json

Each player (by canonical ID e.g. 14, 20, 5) has their stats accumulated
across all clips they appear in.

Merge rules (same as id-switch rules):
  SUM   : total_distance_m, sprint_time_seconds, total_sprint_distance_m,
           jump_count, total_frames_tracked
  MAX   : max_speed_kmh, sprint_speed_kmh, max_acceleration
  WAVG  : avg_speed_kmh  (weighted by total_frames_tracked)
  MIN   : stamina_percentage
  LAST  : current_acceleration (from latest clip)
  CAT   : speed_history (concatenate all samples)

Output structure:
  {
    "total_clips_merged": 100,
    "players": {
      "player_14": { ...merged stats..., "clips_appeared": [1,2,3,...] },
      "player_20": { ... },
      ...
    }
  }

Usage:
  python merge_all_clips.py
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Merge two stat blocks
# ---------------------------------------------------------------------------

def merge_stats(a: dict, b: dict) -> dict:
    merged: dict = {}

    frames_a = a.get('total_frames_tracked', 0)
    frames_b = b.get('total_frames_tracked', 0)
    total_frames = frames_a + frames_b

    # SUM
    for f in ('total_distance_m', 'sprint_time_seconds',
              'total_sprint_distance_m', 'jump_count', 'total_frames_tracked'):
        merged[f] = a.get(f, 0) + b.get(f, 0)

    # MAX
    for f in ('max_speed_kmh', 'sprint_speed_kmh', 'max_acceleration'):
        merged[f] = max(a.get(f, 0), b.get(f, 0))

    # Weighted average avg_speed
    if total_frames > 0:
        merged['avg_speed_kmh'] = (
            a.get('avg_speed_kmh', 0) * frames_a +
            b.get('avg_speed_kmh', 0) * frames_b
        ) / total_frames
    else:
        merged['avg_speed_kmh'] = 0.0

    # MIN stamina
    merged['stamina_percentage'] = min(
        a.get('stamina_percentage', 100),
        b.get('stamina_percentage', 100)
    )

    # LAST current_acceleration (from whichever has more frames = more tracking)
    merged['current_acceleration'] = (
        a.get('current_acceleration', 0)
        if frames_a >= frames_b
        else b.get('current_acceleration', 0)
    )

    # CAT speed_history
    merged['speed_history'] = (
        a.get('speed_history', []) + b.get('speed_history', [])
    )

    # Carry forward clips_appeared list
    merged['clips_appeared'] = sorted(set(
        a.get('clips_appeared', []) + b.get('clips_appeared', [])
    ))

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir   = Path(__file__).parent
    input_dir  = base_dir / 'json_replaced'
    output_file = base_dir / 'merged_players.json'

    print(f"[Input]  {input_dir}")
    print(f"[Output] {output_file}\n")

    json_files = sorted(
        input_dir.glob('*_analysis.json'),
        key=lambda p: int(re.match(r'^(\d+)_', p.name).group(1))
        if re.match(r'^(\d+)_', p.name) else 0
    )

    if not json_files:
        print("No JSON files found in json_replaced/. Run replace_player_ids.py first.")
        return

    # accumulated: { "player_14": {...stats...}, ... }
    accumulated: dict[str, dict] = {}
    clips_processed = 0

    for json_file in json_files:
        m = re.match(r'^(\d+)_', json_file.name)
        clip_num = int(m.group(1)) if m else 0

        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)

        player_stats: dict = data.get('player_stats', {})

        for key, stats in player_stats.items():
            if key == 'match_summary':
                continue
            if not re.match(r'^player_\d+$', key):
                continue

            # Tag which clip this came from
            stats_with_clip = {**stats, 'clips_appeared': [clip_num]}

            if key in accumulated:
                accumulated[key] = merge_stats(accumulated[key], stats_with_clip)
            else:
                accumulated[key] = stats_with_clip

        clips_processed += 1

    # Sort players by ID numerically
    sorted_players = dict(
        sorted(accumulated.items(),
               key=lambda item: int(re.search(r'\d+', item[0]).group()))
    )

    output = {
        "total_clips_merged": clips_processed,
        "total_players": len(sorted_players),
        "players": sorted_players
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Merged {clips_processed} clips.")
    print(f"Total unique players: {len(sorted_players)}")
    print()
    for pkey, pstats in sorted_players.items():
        clips = pstats.get('clips_appeared', [])
        print(f"  {pkey:12s}  clips={len(clips):3d}  "
              f"dist={pstats.get('total_distance_m', 0):8.1f}m  "
              f"max_spd={pstats.get('max_speed_kmh', 0):5.1f}km/h  "
              f"jumps={pstats.get('jump_count', 0)}")
    print(f"\nSaved -> {output_file}")


if __name__ == '__main__':
    main()
