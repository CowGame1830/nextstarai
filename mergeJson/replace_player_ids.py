"""
replace_player_ids.py
=====================
Reads allclip.csv from ./csv/ and rewrites all JSON analysis files in ./json/
with canonical player IDs, saving results to ./json_replaced/.

CSV format:
  video, matetar(14), wharton(20), lacorix(5)
  1,     3,           14,
  2,     14 57,       11 45,       19

  - Column header "matetar(14)" → target ID = 14
  - Cell "14 57" → source IDs 14 and 57 both map to target ID 14
  - If two sources map to same target (id-switch), their stats are merged.

Usage:
  python replace_player_ids.py
"""

import json
import re
import csv
import copy
from pathlib import Path


# ---------------------------------------------------------------------------
# 1.  Parse CSV → per-clip mapping
# ---------------------------------------------------------------------------

def parse_csv_mapping(csv_path: Path) -> dict[int, dict[int, int]]:
    """
    Returns:
        { clip_num: { source_id: target_id, ... }, ... }
    """
    mapping: dict[int, dict[int, int]] = {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)

    # Extract target IDs from headers  e.g. "matetar(14)" → 14
    target_ids: list[int | None] = []
    for h in headers[1:]:
        m = re.search(r'\((\d+)\)', h)
        target_ids.append(int(m.group(1)) if m else None)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                clip_num = int(row[0].strip())
            except ValueError:
                continue

            clip_map: dict[int, int] = {}
            for col_idx, cell in enumerate(row[1:]):
                if col_idx >= len(target_ids):
                    break
                target_id = target_ids[col_idx]
                if target_id is None:
                    continue
                cell = cell.strip()
                if not cell:
                    continue
                # Split on spaces or commas (handles quoted "2,15" cells)
                for tok in re.split(r'[\s,]+', cell):
                    tok = tok.strip()
                    if tok:
                        try:
                            clip_map[int(tok)] = target_id
                        except ValueError:
                            pass

            mapping[clip_num] = clip_map

    return mapping


# ---------------------------------------------------------------------------
# 2.  Merge two player_stat blocks (id-switch: two sources → same target)
# ---------------------------------------------------------------------------

def merge_stats(a: dict, b: dict) -> dict:
    """
    Merge stat block b into a.
    - sum  : distances, times, jump_count, frames_tracked
    - max  : max_speed, sprint_speed, max_acceleration
    - wavg : avg_speed (weighted by frames_tracked)
    - min  : stamina_percentage  (most-tired reading)
    - last : current_acceleration (from block with more frames)
    - cat  : speed_history (concatenate)
    """
    merged: dict = {}

    frames_a = a.get('total_frames_tracked', 0)
    frames_b = b.get('total_frames_tracked', 0)
    total_frames = frames_a + frames_b

    # — sum —
    for f in ('total_distance_m', 'sprint_time_seconds',
              'total_sprint_distance_m', 'jump_count', 'total_frames_tracked'):
        merged[f] = a.get(f, 0) + b.get(f, 0)

    # — max —
    for f in ('max_speed_kmh', 'sprint_speed_kmh', 'max_acceleration'):
        merged[f] = max(a.get(f, 0), b.get(f, 0))

    # — weighted average —
    if total_frames > 0:
        merged['avg_speed_kmh'] = (
            a.get('avg_speed_kmh', 0) * frames_a +
            b.get('avg_speed_kmh', 0) * frames_b
        ) / total_frames
    else:
        merged['avg_speed_kmh'] = 0.0

    # — min stamina —
    merged['stamina_percentage'] = min(
        a.get('stamina_percentage', 100),
        b.get('stamina_percentage', 100)
    )

    # — current_acceleration: keep from whichever block had more tracking time —
    merged['current_acceleration'] = (
        a.get('current_acceleration', 0)
        if frames_a >= frames_b
        else b.get('current_acceleration', 0)
    )

    # — concatenate speed histories —
    merged['speed_history'] = (
        a.get('speed_history', []) + b.get('speed_history', [])
    )

    return merged


# ---------------------------------------------------------------------------
# 3.  Apply mapping to a single JSON dict
# ---------------------------------------------------------------------------

def replace_ids(data: dict, clip_map: dict[int, int], warnings: list[str],
                filename: str) -> dict:
    data = copy.deepcopy(data)
    clip_num_str = filename.split('_')[0]

    # ── detected_player_ids ──────────────────────────────────────────────
    # Positional replacement: each source ID -> target ID, duplicates kept.
    # IDs not in the CSV mapping for this clip are DROPPED.
    # e.g. [10, 13, 21, 36, 55, 57] with 57 unmapped -> [14, 20, 5, 20, 20, 5] (57 dropped)
    new_detected: list[int] = []
    for pid in data.get('detected_player_ids', []):
        if pid not in clip_map:
            warnings.append(
                f"[WARN] {filename}: player_id {pid} not in CSV mapping -> DROPPED"
            )
            continue  # drop it
        new_detected.append(clip_map[pid])
    data['detected_player_ids'] = new_detected

    # ── player_stats ─────────────────────────────────────────────────────
    old_stats: dict = data.get('player_stats', {})
    new_stats: dict = {}

    for key, value in old_stats.items():
        if key == 'match_summary':
            continue  # rebuilt below

        m = re.match(r'^player_(\d+)$', key)
        if not m:
            new_stats[key] = value  # unknown key format — keep as-is
            continue

        src_id = int(m.group(1))
        if src_id not in clip_map:
            continue  # not in CSV for this clip -> DROP the stat block

        tgt_id = clip_map[src_id]
        new_key = f'player_{tgt_id}'

        if new_key in new_stats:
            # id-switch: merge
            new_stats[new_key] = merge_stats(new_stats[new_key], value)
        else:
            new_stats[new_key] = value

    # rebuild match_summary
    unique_real_players = len(
        [k for k in new_stats if re.match(r'^player_\d+$', k)]
    )
    if 'match_summary' in old_stats:
        new_stats['match_summary'] = {
            **old_stats['match_summary'],
            'total_players': unique_real_players,
        }

    data['player_stats'] = new_stats

    # ── total_detected_players ───────────────────────────────────────────
    # Keep the original count (length of detected_player_ids, including duplicates)
    data['total_detected_players'] = len(new_detected)

    # ── selection_history ────────────────────────────────────────────────
    # Same positional replacement; unmapped IDs are dropped.
    for entry in data.get('selection_history', []):
        entry['selected_player_ids'] = [
            clip_map[pid]
            for pid in entry.get('selected_player_ids', [])
            if pid in clip_map
        ]

    return data


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir   = Path(__file__).parent
    csv_path   = base_dir / 'csv' / 'allclip.csv'
    json_dir   = base_dir / 'json'
    output_dir = base_dir / 'json_replaced'
    output_dir.mkdir(exist_ok=True)

    print(f"[CSV]    {csv_path}")
    print(f"[Input]  {json_dir}")
    print(f"[Output] {output_dir}\n")

    mapping = parse_csv_mapping(csv_path)
    print(f"Loaded CSV mappings for {len(mapping)} clips.\n")

    json_files = sorted(json_dir.glob('*_analysis.json'),
                        key=lambda p: int(re.match(r'^(\d+)_', p.name).group(1))
                        if re.match(r'^(\d+)_', p.name) else 0)

    success = 0
    skipped = 0
    warnings: list[str] = []

    for json_file in json_files:
        m = re.match(r'^(\d+)_', json_file.name)
        if not m:
            print(f"  [SKIP] Cannot parse clip number: {json_file.name}")
            skipped += 1
            continue

        clip_num = int(m.group(1))

        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)

        if clip_num not in mapping:
            warnings.append(
                f"[WARN] Clip {clip_num} ({json_file.name}) has no CSV row -> copied unchanged"
            )
            out_path = output_dir / json_file.name
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            skipped += 1
            continue

        clip_map = mapping[clip_num]
        updated  = replace_ids(data, clip_map, warnings, json_file.name)

        out_path = output_dir / json_file.name
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)

        # Pretty summary line
        old_ids = data.get('detected_player_ids', [])
        new_ids = updated.get('detected_player_ids', [])
        print(f"  [OK] {json_file.name:50s}  {old_ids} -> {new_ids}")
        success += 1

    print(f"\nDone -- {success} files replaced, {skipped} skipped/copied unchanged.")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")


if __name__ == '__main__':
    main()
