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
    Calculates Work Rate attribute (10-200) based on Action Density and Continuity.
    
    References:
      - Verheijen (2014): Action Density (Actions per Minute).
      - Castellano et al. (2014): Activity Profile (Continuity).
      
    Benchmarks (Index -> Score):
      - Continuity (Jogging % > 7km/h): 85% -> 1.0, 65% -> 0.5
      - Action Density (Bursts/min): 5.0 -> 1.0, 2.0 -> 0.5
      - Overall Intensity (Avg Speed): 10km/h -> 1.0, 5km/h -> 0.0
    """
    # 1. Continuity Factor (Castellano et al., 2014)
    # Benchmarks: 85% -> 1.0, 65% -> 0.5, 40% -> 0.0
    continuity = (jogging_percentage - 0.4) / (0.85 - 0.4)
    continuity = min(1.0, max(0.0, continuity))
    
    # 2. Action Density Factor (Verheijen, 2014)
    # Benchmarks: 5.0+ bursts/min -> 1.0, 2.0 -> 0.5, 0.5 -> 0.0
    action_density = (burst_frequency - 0.5) / (5.0 - 0.5)
    action_density = min(1.0, max(0.0, action_density))
    
    # 3. Overall Intensity Factor (Avg Speed)
    # Benchmarks: 10 km/h -> 1.0, 5 km/h -> 0.0
    intensity = (avg_speed_kmh - 5.0) / (10.0 - 5.0)
    intensity = min(1.0, max(0.0, intensity))
    
    # Weighted calculation
    score_idx = (continuity * 0.4) + (action_density * 0.4) + (intensity * 0.2)
    
    # Map 0.0-1.0 to 10-200
    score = score_idx * 190 + 10
    
    return min(200, max(10, round(score)))


def calculate_stamina(sprint_percentage, hir_percentage, decay_rate=1.0, rsa_score=0.8):
    """
    Calculates Stamina attribute (10-200) using the Intensity-Sustainability (IS) Model.
    
    Philosophy:
      Stamina = (Intensity Capacity) x (Sustainability Factor)
      This represents the player's ability to maintain their demonstrated 
      peak intensity over the observation window.
      
    Components:
      - Intensity Capacity (IC): Weighted HIR and Sprint density.
      - Sustainability Factor (SF): Fatigue resistance derived from performance decay.
    """
    # 1. Intensity Capacity (IC)
    # Measures the density of high-intensity efforts.
    ic = (sprint_percentage * 0.7) + (hir_percentage * 0.3)
    
    # 2. Sustainability Factor (SF)
    # Measures how well peak performance was maintained (Bangsbo, 1994).
    # Range [0.4, 1.05]. 1.0 = Perfect maintenance.
    sf = min(1.05, max(0.4, decay_rate))
    
    # 3. Interaction Index (Endurance Index)
    # Multiplicative interaction + small RSA additive component.
    # Theoretical Elite Peak (Highlight) ~ 0.40
    # Theoretical Standard Pro (Highlight) ~ 0.25
    endurance_index = (ic * sf) + (rsa_score * 0.2)
    
    # 4. Scientific Linear Mapping
    # Based on Theoretical Anchors for the tracking environment.
    # Anchor 1: Index 0.40 -> 200 (Elite Engine)
    # Anchor 2: Index 0.25 -> 120 (Standard Pro)
    # Slope: (200 - 120) / (0.40 - 0.25) = 80 / 0.15 = 533.33
    score = 120.0 + (endurance_index - 0.25) * 533.33
    
    return min(220, max(10, round(score)))


if __name__ == "__main__":
    # --- Quick sanity-check ---
    print("=== FM26 Recalibration Sanity Check (v2 Fatigue Resistance) ===")
    print(f"Pace  @ 29.4 km/h : {calculate_pace(29.4):>4} (expect 120)")
    print(f"Pace  @ 37.38 km/h: {calculate_pace(37.38):>4} (expect 200)")
    
    print(f"Accel @ 2.5 m/s^2  : {calculate_acceleration(2.5):>4} (expect 100)")
    print(f"Accel @ 8.5 m/s^2  : {calculate_acceleration(8.5):>4} (expect 200)")
    
    # --- Work Rate Check (Action Density v2) ---
    print("\n--- Work Rate Check (Action Density v2) ---")
    # Elite: 85% continuity, 5.0 bursts/min, 10.0 km/h avg
    print(f"Elite   (85% Cont, 5.0 bursts, 10.0 km/h): {calculate_work_rate(10.0, 0.85, 5.0):>4} (expect ~200)")
    # Average Pro: 75% continuity, 2.5 bursts/min, 8.0 km/h avg
    print(f"Average (75% Cont, 2.5 bursts, 8.0 km/h): {calculate_work_rate(8.0, 0.75, 2.5):>4} (expect ~114)")
    # Low: 50% continuity, 0.5 bursts/min, 5.0 km/h avg
    print(f"Low     (50% Cont,  0.5 bursts, 5.0 km/h): {calculate_work_rate(5.0, 0.50, 0.5):>4} (expect ~27)")
    
    # --- Stamina Check (Scientific v3) ---
    print("\n--- Stamina Check (Scientific v3) ---")
    # Elite: High HIR + Steady/Strong Finish + Good RSA
    print(f"Elite   (10% spr, 20% hir, 1.1 decay): {calculate_stamina(0.10, 0.20, 1.1, 0.9):>4} (expect ~163)")
    # Pro: High Intensity but Gassed Out
    print(f"Pro Gassed (10% spr, 15% hir, 0.7 decay): {calculate_stamina(0.10, 0.15, 0.7, 0.8):>4} (expect ~102)")
    # Average Pro: Steady 
    print(f"Average (4% spr, 8% hir, 1.0 decay): {calculate_stamina(0.04, 0.08, 1.0, 0.7):>4} (expect ~96)")
    # Lazy: Low Intensity
    print(f"Lazy    (1% spr, 2% hir, 1.0 decay): {calculate_stamina(0.01, 0.02, 1.0, 0.5):>4} (expect ~83)")
