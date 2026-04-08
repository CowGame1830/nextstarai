import json

# ============================================================
#  High-Precision Football Attributes (10–200 scale)
#  Based on FM 1-20 scale, expanded to 10-200.
#  All benchmarks are world-class professional references.
# ============================================================

def calculate_pace(max_speed_kmh):
    """
    Calculates Pace attribute (10-200) based on top speed.

    World-class benchmarks:
      - 22 km/h  →  10  (minimum)
      - 37 km/h  → 200  (elite world-class)

    Formula: ((max_speed_kmh - 22) / 15) * 190 + 10
    """
    score = ((max_speed_kmh - 22) / 15) * 190 + 10
    return min(200, max(10, round(score)))


def calculate_acceleration(max_acceleration_ms2):
    """
    Calculates Acceleration attribute (10-200) based on peak acceleration.

    World-class benchmarks:
      -  2 m/s²  →  10  (minimum)
      - 12 m/s²  → 200  (elite world-class)

    Formula: ((max_acceleration_ms2 - 2) / 10) * 190 + 10
    """
    score = ((max_acceleration_ms2 - 2) / 10) * 190 + 10
    return min(200, max(10, round(score)))


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
    print("=== Sanity Check ===")
    print(f"Pace  @ 22 km/h  : {calculate_pace(22):>4}  (expect 10)")
    print(f"Pace  @ 37 km/h  : {calculate_pace(37):>4}  (expect 200)")
    print(f"Pace  @ 30 km/h  : {calculate_pace(30):>4}  (expect ~101)")
    print(f"Accel @  2 m/s²  : {calculate_acceleration(2):>4}  (expect 10)")
    print(f"Accel @ 12 m/s²  : {calculate_acceleration(12):>4}  (expect 200)")
    print(f"Accel @  7 m/s²  : {calculate_acceleration(7):>4}  (expect ~105)")
    print(f"Work  avg=10 spr=0.075: {calculate_work_rate(10, 0.075):>4}  (expect 105)")
    print(f"Work  avg=20 spr=0.15 : {calculate_work_rate(20, 0.15):>4}  (expect 200)")
    print(f"Stam  @   0%     : {calculate_stamina(0):>4}  (expect 10)")
    print(f"Stam  @  50%     : {calculate_stamina(50):>4}  (expect ~105)")
    print(f"Stam  @ 100%     : {calculate_stamina(100):>4}  (expect 200)")
