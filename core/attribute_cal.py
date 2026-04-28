import json

# ============================================================
#  High-Precision Football Attributes (10–200 scale)
#  Based on FM 1-20 scale, expanded to 10-200.
#  All benchmarks are world-class professional references.
# ============================================================

def calculate_pace(max_speed_kmh):
    """
    Calculates Pace attribute (10-200) based on top speed.

    FM26 Benchmarks:
      - 29.4 km/h  →  120
      - 37.38 km/h →  200

    Formula: Linear scale between 29.4 and 37.38 km/h.
    """
    # Slope: (200 - 120) / (37.38 - 29.4) = 80 / 7.98
    slope = 80.0 / 7.98
    score = 120.0 + (max_speed_kmh - 29.4) * slope
    
    return min(220, max(10, round(score)))


def calculate_acceleration(max_acceleration_ms2):
    """
    Calculates Acceleration attribute (10-200) based on peak burst.

    FM26 Benchmarks:
      - 2.5 m/s²  →  100 (Standard Start, FM 10)
      - 6.5 m/s²  →  180 (Elite Burst, FM 18 - e.g. Traoré)
      - 8.5 m/s²  →  200 (Olympic Burst, FM 20)

    Formula: Piecewise linear scale.
    """
    if max_acceleration_ms2 < 6.5:
        # Segment 1: [2.5, 6.5] -> [100, 180]
        # Slope: (180 - 100) / (6.5 - 2.5) = 80 / 4 = 20
        score = 100.0 + (max_acceleration_ms2 - 2.5) * 20.0
    else:
        # Segment 2: [6.5, 8.5] -> [180, 200]
        # Slope: (200 - 180) / (8.5 - 6.5) = 20 / 2 = 10
        score = 180.0 + (max_acceleration_ms2 - 6.5) * 10.0
        
    return min(220, max(10, round(score)))


def calculate_work_rate(avg_speed_kmh, jogging_percentage, burst_frequency):
    """
    Calculates Work Rate attribute (10-200) based on Action Density.
    
    Work Rate is about "Willingness" - constant movement and pressing.
    
    Benchmarks (Index -> Score):
      - Continuity:  85% -> 1.0, 65% -> 0.5
      - Burst Freq:  5.0 -> 1.0, 2.0 -> 0.5
      - Intensity:  10km -> 1.0, 5km -> 0.0
    """
    # 1. Continuity Factor (Jogging %)
    # Benchmarks: 85% -> 1.0, 65% -> 0.5, 40% -> 0.0
    continuity = (jogging_percentage - 0.4) / (0.85 - 0.4)
    continuity = min(1.0, max(0.0, continuity))
    
    # 2. Burst Frequency Factor (Starts per minute)
    # Benchmarks: 5.0+ -> 1.0, 2.0 -> 0.5, 0.5 -> 0.0
    burst_f = (burst_frequency - 0.5) / (5.0 - 0.5)
    burst_f = min(1.0, max(0.0, burst_f))
    
    # 3. Intensity Factor (Avg Speed)
    # Benchmarks: 10 km/h -> 1.0, 5 km/h -> 0.0
    intensity = (avg_speed_kmh - 5.0) / (10.0 - 5.0)
    intensity = min(1.0, max(0.0, intensity))
    
    score_idx = (continuity * 0.4) + (burst_f * 0.4) + (intensity * 0.2)
    
    # Map 0.0-1.0 to 10-200
    score = score_idx * 190 + 10
    
    return min(200, max(10, round(score)))


def calculate_stamina(avg_speed_kmh, sprint_percentage, decay_rate=1.0):
    """
    Calculates Stamina attribute (10-200) based on Endurance Capacity (Fatigue Resistance).
    
    Formula: Endurance_Index = (Work_Volume) * Clamped_Decay
    - Work_Volume: Intensity proxy (Avg Speed + Sprint Intensity)
    - Decay_Rate: Fatigue resistance proxy (Peak performance at end vs start)
    
    Benchmarks (Index -> Score):
      - 21.0  -> 200 (World Class - High Intensity + Strong Finish)
      - 12.0  -> 110 (Professional - Steady Performance)
      - 5.0   -> 40  (Low - Amateur)
    """
    # Clamp decay_rate to avoid outliers in short samples [0.7, 1.2]
    clamped_decay = min(1.2, max(0.7, decay_rate))
    
    work_volume = avg_speed_kmh + (sprint_percentage * 50.0)
    endurance_index = work_volume * clamped_decay
    
    if endurance_index >= 12.0:
        # Segment: [12, 21] -> [110, 200]
        # Slope: 90 / 9 = 10.0
        score = 110.0 + (endurance_index - 12.0) * 10.0
    else:
        # Segment: [5, 12] -> [40, 110]
        # Slope: 70 / 7 = 10.0
        score = 40.0 + (endurance_index - 5.0) * 10.0
        
    return min(220, max(10, round(score)))


if __name__ == "__main__":
    # --- Quick sanity-check ---
    print("=== FM26 Recalibration Sanity Check (v2 Fatigue Resistance) ===")
    print(f"Pace  @ 29.4 km/h : {calculate_pace(29.4):>4} (expect 120)")
    print(f"Pace  @ 37.38 km/h: {calculate_pace(37.38):>4} (expect 200)")
    
    print(f"Accel @ 2.5 m/s²  : {calculate_acceleration(2.5):>4} (expect 100)")
    print(f"Accel @ 8.5 m/s²  : {calculate_acceleration(8.5):>4} (expect 200)")
    
    # --- Work Rate Check (Action Density v2) ---
    print("\n--- Work Rate Check (Action Density v2) ---")
    # Elite: 85% continuity, 5.0 bursts/min, 10.0 km/h avg
    print(f"Elite   (85% Cont, 5.0 bursts, 10.0 km/h): {calculate_work_rate(10.0, 0.85, 5.0):>4} (expect ~200)")
    # Average Pro: 75% continuity, 2.5 bursts/min, 8.0 km/h avg
    print(f"Average (75% Cont, 2.5 bursts, 8.0 km/h): {calculate_work_rate(8.0, 0.75, 2.5):>4} (expect ~114)")
    # Low: 50% continuity, 0.5 bursts/min, 5.0 km/h avg
    print(f"Low     (50% Cont,  0.5 bursts, 5.0 km/h): {calculate_work_rate(5.0, 0.50, 0.5):>4} (expect ~27)")
    
    # --- Stamina Check (Endurance Capacity v2) ---
    print("\n--- Stamina Check (Endurance Capacity v2) ---")
    # Elite: High Intensity + Steady/Strong Finish
    print(f"Elite   (11 km/h, 10% spr, 1.1 decay): {calculate_stamina(11.0, 0.10, 1.1):>4} (expect ~166)")
    # Pro: High Intensity but Gassed Out
    print(f"Pro Gassed (11 km/h, 10% spr, 0.7 decay): {calculate_stamina(11.0, 0.10, 0.7):>4} (expect ~102)")
    # Average Pro: Steady 
    print(f"Average ( 8 km/h,  4% spr, 1.0 decay): {calculate_stamina(8.0, 0.04, 1.0):>4} (expect ~90)")
    # Lazy: Low Intensity
    print(f"Lazy    ( 4.5 km/h, 1% spr, 1.0 decay): {calculate_stamina(4.5, 0.01, 1.0):>4} (expect ~40)")
