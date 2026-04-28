import json
from pathlib import Path

def get_percentile(arr, p):
    if not arr: return 0
    sorted_arr = sorted(arr)
    idx = int(len(sorted_arr) * (p / 100))
    return sorted_arr[min(idx, len(sorted_arr) - 1)]

def analyze_work_rate_signals(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    players = data.get('players', {})
    stats_summary = []

    for pid, pdata in players.items():
        history = pdata.get('speed_history', [])
        if len(history) < 100: continue
        
        n = len(history)
        total_seconds = n / 30.0
        
        # 1. Jogging % (> 7 km/h)
        jog_frames = len([s for s in history if s > 7.0])
        jog_pct = jog_frames / n
        
        # 2. Burst Frequency (Sprint starts per minute)
        # Refined: Must stay above 15.0 for at least 10 frames to count as a real burst
        bursts = 0
        in_burst = False
        burst_duration = 0
        
        for i in range(len(history)):
            if history[i] >= 15.0:
                burst_duration += 1
                if not in_burst and burst_duration >= 10: # Stayed above for 1/3 second
                    bursts += 1
                    in_burst = True
            else:
                in_burst = False
                burst_duration = 0
        
        burst_freq = bursts / (total_seconds / 60.0) if total_seconds > 0 else 0
        
        avg_speed = pdata.get('avg_speed_kmh', 0)
        
        stats_summary.append({
            'id': pid,
            'avg_speed': avg_speed,
            'jog_pct': jog_pct,
            'burst_freq': burst_freq,
            'samples': n
        })
    
    return stats_summary

if __name__ == "__main__":
    path = Path(r"d:\nextstarAI\attribute_calculate\mergeJson\merged_players.json")
    if not path.exists():
        print(f"File not found: {path}")
    else:
        results = analyze_work_rate_signals(path)
        
        print(f"{'Player':<12} | {'Avg Spd':>8} | {'Jog%':>8} | {'Burst/Min':>10} | {'Samples':>8}")
        print("-" * 65)
        for r in results:
            print(f"{r['id']:<12} | {r['avg_speed']:>8.2f} | {r['jog_pct']:>8.1%} | {r['burst_freq']:>10.2f} | {r['samples']:>8}")

        if results:
            jog_pcts = [r['jog_pct'] for r in results]
            burst_freqs = [r['burst_freq'] for r in results]
            
            print("\n--- Percentiles for Benchmarking ---")
            for name, arr in [("Jog %", jog_pcts), ("Burst Freq", burst_freqs)]:
                print(f"{name:<12}: P95={get_percentile(arr, 95):.2f}, P50={get_percentile(arr, 50):.2f}, P5={get_percentile(arr, 5):.2f}")
