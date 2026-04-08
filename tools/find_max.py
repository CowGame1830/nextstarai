import json

with open('input/player_stats_20251119_133015.json', 'r') as f:
    data = json.load(f)

max_speed = 0
max_acc = 0
max_dist_per_frame = 0

for key, p in data.items():
    if key == "match_summary":
        continue
    
    max_speed = max(max_speed, p.get('max_speed_kmh', 0))
    max_acc = max(max_acc, p.get('max_acceleration', 0))
    
    frames = p.get('total_frames_tracked', 0)
    if frames > 0:
        max_dist_per_frame = max(max_dist_per_frame, p.get('total_distance_m', 0) / frames)

print(f"MAX Speed: {max_speed}")
print(f"MAX Acceleration: {max_acc}")
print(f"MAX Distance per frame: {max_dist_per_frame}")
