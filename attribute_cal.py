import json
import math

def calculate_pace(max_speed_kmh, dataset_max_speed=355.5):
    """
    Calculates Pace attribute (1-20) based on top speed.
    """
    if dataset_max_speed == 0: return 1
    # Scale: 1-20. We use a non-linear scaling (square root) to differentiate speed better
    # But for simplicity, a linear mapping normalized to the dataset max is often preferred.
    score = (max_speed_kmh / dataset_max_speed) * 19 + 1
    return min(20, max(1, round(score)))

def calculate_acceleration(max_acc, dataset_max_acc=7139.0):
    """
    Calculates Acceleration attribute (1-20) based on peak acceleration.
    """
    if dataset_max_acc == 0: return 1
    score = (max_acc / dataset_max_acc) * 19 + 1
    return min(20, max(1, round(score)))

def calculate_stamina(stamina_percentage, total_distance_m, total_frames_tracked):
    """
    Calculates Stamina attribute (1-20).
    Combines tracking percentage with physical output (distance per frame).
    """
    if total_frames_tracked == 0: return 1
    
    # physical_output_score: how much they moved relative to time
    # Max distance per frame in dataset is ~35.8
    dist_per_frame = total_distance_m / total_frames_tracked
    physical_score = min(1.0, dist_per_frame / 35.0)
    
    # Combine with stamina_percentage (already 0-100)
    normalized_stamina = stamina_percentage / 100.0
    
    # Average of both factors
    score = ((physical_score + normalized_stamina) / 2) * 19 + 1
    return min(20, max(1, round(score)))

def calculate_work_rate(total_distance_m, total_frames_tracked, sprint_time_seconds):
    """
    Calculates Work Rate attribute (1-20).
    Based on movement intensity and sprint frequency.
    """
    if total_frames_tracked == 0: return 1
    
    # movement_intensity: distance per frame
    intensity = min(1.0, (total_distance_m / total_frames_tracked) / 25.0)
    
    # sprint_frequency: sprint time relative to total tracking time
    # Assuming 30fps, total seconds = frames / 30
    total_seconds = total_frames_tracked / 30.0
    if total_seconds > 0:
        sprint_ratio = min(1.0, (sprint_time_seconds / total_seconds) / 0.8) # 80% sprint time is elite
    else:
        sprint_ratio = 0
        
    score = ((intensity + sprint_ratio) / 2) * 19 + 1
    return min(20, max(1, round(score)))

if __name__ == "__main__":
    # Test with sample JSON
    with open('input/player_stats_20251119_133015.json', 'r') as f:
        data = json.load(f)
    
    print(f"{'Player':<10} | {'Pace':<5} | {'Acc':<5} | {'Stam':<5} | {'Work':<5}")
    print("-" * 45)
    
    for key, p in data.items():
        if key == "match_summary": continue
        
        pace = calculate_pace(p.get('max_speed_kmh', 0))
        acc = calculate_acceleration(p.get('max_acceleration', 0))
        stam = calculate_stamina(p.get('stamina_percentage', 0), p.get('total_distance_m', 0), p.get('total_frames_tracked', 0))
        wr = calculate_work_rate(p.get('total_distance_m', 0), p.get('total_frames_tracked', 0), p.get('sprint_time_seconds', 0))
        
        print(f"{key:<10} | {pace:<5} | {acc:<5} | {stam:<5} | {wr:<5}")
