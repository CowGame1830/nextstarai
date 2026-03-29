import json
from attribute_cal import calculate_pace, calculate_acceleration, calculate_stamina, calculate_work_rate

def main():
    json_path = 'input/player_stats_20251119_133015.json'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Print Header
    print(f"{'Player ID':<12} | {'Pace':<6} | {'Acc':<6} | {'Stam':<6} | {'Work R':<6}")
    print("-" * 55)
    
    players = []
    for key, stats in data.items():
        if key == "match_summary":
            continue
            
        # Extract ID from key (e.g., 'player_1' -> 1)
        player_id = key.split('_')[1]
        
        # Calculate Attributes
        pace = calculate_pace(stats.get('max_speed_kmh', 0))
        acc = calculate_acceleration(stats.get('max_acceleration', 0))
        stam = calculate_stamina(
            stats.get('stamina_percentage', 0), 
            stats.get('total_distance_m', 0), 
            stats.get('total_frames_tracked', 0)
        )
        wr = calculate_work_rate(
            stats.get('total_distance_m', 0), 
            stats.get('total_frames_tracked', 0), 
            stats.get('sprint_time_seconds', 0)
        )
        
        # Store for sorting or direct print
        players.append({
            'id': player_id,
            'pace': pace,
            'acc': acc,
            'stam': stam,
            'wr': wr
        })
    
    # Sort by ID numerically
    players.sort(key=lambda x: int(x['id']))
    
    # Print Rows
    for p in players:
        # Hide players with 0 stats (optional, but cleaner)
        if p['pace'] == 1 and p['acc'] == 1 and p['stam'] == 1 and p['wr'] == 1:
            continue
            
        print(f"Player {p['id']:<5} | {p['pace']:<6} | {p['acc']:<6} | {p['stam']:<6} | {p['wr']:<6}")

if __name__ == "__main__":
    main()
