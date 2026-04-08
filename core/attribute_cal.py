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
      - 24 km/h    →  100 (Standard Pro, FM 10)
      - 37.4 km/h  →  200 (Van de Ven Record, FM 20)

    Formula: Linear scale between 24 and 37.4 km/h.
    """
    # Slope: (200 - 100) / (37.4 - 24.0) = 100 / 13.4
    slope = 100.0 / 13.4
    score = 100.0 + (max_speed_kmh - 24.0) * slope
    
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


def calculate_work_rate(avg_speed_kmh, sprint_percentage):
    """
    Calculates Work Rate attribute (10-200) based on effort and intensity.

    World-class benchmarks:
      - Average Speed: 20 km/h baseline for elite intensity
      - Sprint Percentage: 15% of total time as elite frequency

    Inputs:
      avg_speed_kmh    - Average movement speed during the match (km/h)
      sprint_percentage - Proportion of time spent sprinting (0.0 – 1.0)
                          e.g. 15% → 0.15

    Intensity  = min(1.0, avg_speed_kmh / 20.0)
    Frequency  = min(1.0, sprint_percentage / 0.15)
    Formula    = ((intensity + frequency) / 2) * 190 + 10
    """
    intensity  = min(1.0, avg_speed_kmh / 20.0)
    frequency  = min(1.0, sprint_percentage / 0.15)
    score = ((intensity + frequency) / 2) * 190 + 10
    return min(200, max(10, round(score)))


def calculate_stamina(stamina_percentage):
    """
    Calculates Stamina attribute (10-200) based on energy remaining.

    Direct linear mapping (0-100% → 10-200):
      - 0%   →  10
      - 50%  → 105
      - 100% → 200

    Formula: (stamina_percentage / 100) * 190 + 10
    """
    score = (stamina_percentage / 100) * 190 + 10
    return min(200, max(10, round(score)))


if __name__ == "__main__":
    # --- Quick sanity-check ---
    print("=== FM26 Recalibration Sanity Check ===")
    print(f"Pace  @ 24.0 km/h : {calculate_pace(24.0):>4} (expect 100)")
    print(f"Pace  @ 37.4 km/h : {calculate_pace(37.4):>4} (expect 200)")
    print(f"Pace  @ 30.0 km/h : {calculate_pace(30.0):>4} (expect ~145)")
    
    print(f"Accel @ 2.5 m/s²  : {calculate_acceleration(2.5):>4} (expect 100)")
    print(f"Accel @ 6.5 m/s²  : {calculate_acceleration(6.5):>4} (expect 180)")
    print(f"Accel @ 8.5 m/s²  : {calculate_acceleration(8.5):>4} (expect 200)")
    
    # Check Work Rate & Stamina (unchanged)
    print(f"Work Intensity=1.0 freq=1.0: {calculate_work_rate(20, 0.15):>4} (expect 200)")
    print(f"Stamina @ 50%: {calculate_stamina(50):>4} (expect 105)")
