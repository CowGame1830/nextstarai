def get_unified_stats(json_data):
    """
    Unifies player statistics across all checkpoints in a merged_checkpoints JSON.

    IMPORTANT: All player IDs in the file are assumed to be the SAME physical
    person (ID-switching occurs when the tracker loses and reassigns the player).
    All data is merged into one single profile.

    Returns:
        dict — a single unified profile with all fields needed by attribute_cal.py.
    """
    checkpoints = json_data.get("checkpoints", [])
    if not checkpoints:
        return {}

    unified = {
        # Summation
        "total_distance_m":          0.0,
        "jump_count":                0,
        "sprint_time_seconds":       0.0,
        "total_sprint_distance_m":   0.0,
        "total_frames_tracked":      0,
        # Peak values
        "max_speed_kmh":             0.0,
        "max_acceleration_enhanced": 0.0,  # from enhanced_stats (m/s²)
        # Weighted avg helpers
        "_sum_avg_speed_frames":     0.0,
        # Latest state (last checkpoint wins)
        "stamina_percentage":        0.0,
        "final_stamina_percentage":  0.0,
        # History
        "speed_history":  [],
        "jumps_detected": [],
    }

    seen_checkpoints = set()

    for checkpoint in checkpoints:
        cp_name = checkpoint.get("checkpoint_name", "")
        player_stats_dict   = checkpoint.get("player_stats", {})
        enhanced_stats_dict = checkpoint.get("enhanced_stats", {})

        # Each checkpoint has one active player key (e.g. 'player_13' or 'player_66')
        player_key   = next((k for k in player_stats_dict  if k.startswith("player_")), None)
        enhanced_key = next((k for k in enhanced_stats_dict if k.startswith("player_")), None)

        if not player_key:
            continue

        p_stats = player_stats_dict[player_key]
        e_stats = enhanced_stats_dict.get(enhanced_key, {}) if enhanced_key else {}

        frames = p_stats.get("total_frames_tracked", 0)

        # --- Summation ---
        unified["total_distance_m"]        += p_stats.get("total_distance_m", 0.0)
        unified["jump_count"]              += p_stats.get("jump_count", 0)
        unified["sprint_time_seconds"]     += p_stats.get("sprint_time_seconds", 0.0)
        unified["total_sprint_distance_m"] += p_stats.get("total_sprint_distance_m", 0.0)
        unified["total_frames_tracked"]    += frames

        # --- Peak values ---
        unified["max_speed_kmh"] = max(
            unified["max_speed_kmh"], p_stats.get("max_speed_kmh", 0.0)
        )
        # Use enhanced_stats for physics-correct acceleration (m/s²)
        unified["max_acceleration_enhanced"] = max(
            unified["max_acceleration_enhanced"],
            e_stats.get("max_acceleration", 0.0)
        )

        # --- Weighted average speed ---
        avg_speed = p_stats.get("avg_speed_kmh", 0.0)
        unified["_sum_avg_speed_frames"] += avg_speed * frames

        # --- Latest state ---
        unified["stamina_percentage"]       = p_stats.get("stamina_percentage", 0.0)
        unified["final_stamina_percentage"] = e_stats.get("final_stamina_percentage", 0.0)

        # --- History ---
        unified["speed_history"].extend(e_stats.get("speed_history", []))
        unified["jumps_detected"].extend(e_stats.get("jumps_detected", []))

    # --- Finalize ---
    total_frames = unified["total_frames_tracked"]
    speed_history = unified["speed_history"]
    
    if total_frames > 0:
        unified["avg_speed_kmh"] = unified["_sum_avg_speed_frames"] / total_frames
    else:
        unified["avg_speed_kmh"] = 0.0

    # HIR thresholds (Bradley et al., 2009)
    # HIR > 19.8 km/h, Sprint > 25.2 km/h
    if speed_history:
        hir_frames = sum(1 for s in speed_history if s > 19.8)
        unified["hir_percentage"] = hir_frames / len(speed_history)
        
        # Jogging/Moving percentage (> 7 km/h as per Bangsbo)
        jog_frames = sum(1 for s in speed_history if s > 7.0)
        unified["jogging_percentage"] = jog_frames / len(speed_history)
        
        # Robust Decay Rate (compare last 25% of samples to first 25%)
        n = len(speed_history)
        if n >= 4:
            first_q = speed_history[:n//4]
            last_q  = speed_history[-(n//4):]
            avg_first = sum(first_q) / len(first_q)
            avg_last  = sum(last_q) / len(last_q)
            if avg_first > 0:
                unified["decay_rate"] = avg_last / avg_first
            else:
                unified["decay_rate"] = 1.0
        else:
            unified["decay_rate"] = 1.0
            
        # Burst Frequency (acceleration bursts detected from speed diffs)
        # Assuming speed_history points are roughly 1.5s apart based on sample
        bursts = 0
        for i in range(1, len(speed_history)):
            if (speed_history[i] - speed_history[i-1]) > 5.0: # Burst of >5km/h
                bursts += 1
        total_minutes = (total_frames / 30.0) / 60.0
        unified["burst_frequency"] = bursts / total_minutes if total_minutes > 0 else 0
    else:
        unified["hir_percentage"] = 0.0
        unified["jogging_percentage"] = 0.0
        unified["decay_rate"] = 1.0
        unified["burst_frequency"] = 0.0

    # Sprint percentage: fraction of total time spent sprinting (0.0–1.0)
    total_seconds = total_frames / 30.0  # assuming 30 fps
    if total_seconds > 0:
        unified["sprint_percentage"] = min(1.0, unified["sprint_time_seconds"] / total_seconds)
    else:
        unified["sprint_percentage"] = 0.0

    del unified["_sum_avg_speed_frames"]

    return unified  # single dict for one physical player
