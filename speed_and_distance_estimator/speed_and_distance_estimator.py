import cv2
import json
import os
import sys
from collections import deque

import numpy as np

sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
        self.speed_smoothing_alpha = 0.35
        self.max_plausible_speed_kmh = 42.0
        # Enhanced tracking for real-time stats
        self.player_positions_history = {}
        self.player_speeds_history = {}
        self.player_smoothed_speeds = {}
        self.player_accelerations = {}  # Maximum acceleration
        self.player_current_accelerations = {}  # Current acceleration
        self.player_jump_counts = {}
        self.player_stamina = {}
        self.player_sprint_speeds = {}
        self.player_total_distances = {}
        self.player_status = {}  # Current player status (waiting/jogging/running/sprinting)
        self.all_detected_player_ids = set()
        
        # Speed thresholds for status determination (km/h)
        self.status_thresholds = {
            'waiting': 3.0,      # 0-3 km/h
            'jogging': 12.0,     # 3-12 km/h  
            'running': 20.0,     # 12-20 km/h
            'sprinting': float('inf')  # 20+ km/h
        }
        
        self.video_frames = []  # Store frames for jump detection processing
    
    def set_video_frames(self, frames):
        """Store video frames for advanced jump detection processing"""
        self.video_frames = frames

    def _ensure_player_initialized(self, track_id):
        """Ensure all player stat containers exist for a track id."""
        self.all_detected_player_ids.add(track_id)
        if track_id not in self.player_speeds_history:
            self.player_speeds_history[track_id] = deque(maxlen=10)
            self.player_smoothed_speeds[track_id] = 0
            self.player_accelerations[track_id] = 0
            self.player_current_accelerations[track_id] = 0
            self.player_jump_counts[track_id] = 0
            self.player_stamina[track_id] = 100
            self.player_sprint_speeds[track_id] = 0
            self.player_total_distances[track_id] = 0
            self.player_positions_history[track_id] = deque(maxlen=5)
            self.player_status[track_id] = 'waiting'

    def _compute_window_speed(self, object_tracks, track_id, frame_num, last_frame):
        """Compute speed from step-by-step path distance inside the window."""
        valid_points = []
        for idx in range(frame_num, last_frame + 1):
            if track_id not in object_tracks[idx]:
                continue
            pt = object_tracks[idx][track_id].get('position_transformed')
            if pt is None:
                continue
            valid_points.append((idx, pt))

        if len(valid_points) < 2:
            return None

        distance_covered = 0.0
        for i in range(1, len(valid_points)):
            distance_covered += measure_distance(valid_points[i - 1][1], valid_points[i][1])

        start_idx = valid_points[0][0]
        end_idx = valid_points[-1][0]
        if end_idx <= start_idx:
            return None

        time_elapsed = (end_idx - start_idx) / self.frame_rate
        if time_elapsed <= 0:
            return None

        raw_speed_km_per_hour = (distance_covered / time_elapsed) * 3.6
        clipped_speed_km_per_hour = min(raw_speed_km_per_hour, self.max_plausible_speed_kmh)

        previous_smoothed = self.player_smoothed_speeds.get(track_id, clipped_speed_km_per_hour)
        smoothed_speed_km_per_hour = (
            self.speed_smoothing_alpha * clipped_speed_km_per_hour
            + (1 - self.speed_smoothing_alpha) * previous_smoothed
        )
        self.player_smoothed_speeds[track_id] = smoothed_speed_km_per_hour

        return {
            'speed_kmh': smoothed_speed_km_per_hour,
            'distance_m': distance_covered,
            'start_idx': start_idx,
            'end_idx': end_idx,
        }
    
    def add_speed_and_distance_to_tracks(self,tracks):
        # Register all detected players first so they are always included in stats export.
        for frame_players in tracks.get("players", []):
            for track_id in frame_players.keys():
                self._ensure_player_initialized(track_id)

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    self._ensure_player_initialized(track_id)
                    window_metrics = self._compute_window_speed(object_tracks, track_id, frame_num, last_frame)
                    if window_metrics is None:
                        continue

                    distance_covered = window_metrics['distance_m']
                    speed_km_per_hour = window_metrics['speed_kmh']
                    start_idx = window_metrics['start_idx']
                    end_idx = window_metrics['end_idx']

                    # Use first/last valid points in the window for acceleration/jump context.
                    start_position = object_tracks[start_idx][track_id]['position_transformed']
                    end_position = object_tracks[end_idx][track_id]['position_transformed']

                    # Enhanced tracking for real-time stats
                    self._update_player_stats(track_id, speed_km_per_hour, distance_covered, 
                                            start_position, end_position, frame_num)

                    for frame_num_batch in range(start_idx, end_idx + 1):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        track_info = tracks[object][frame_num_batch][track_id]
                        track_info['speed'] = speed_km_per_hour
                        track_info['distance'] = self.player_total_distances[track_id]
                        # Add enhanced stats to tracks
                        track_info['sprint_speed'] = self.player_sprint_speeds.get(track_id, 0)
                        track_info['acceleration'] = self.player_current_accelerations.get(track_id, 0)
                        track_info['max_acceleration'] = self.player_accelerations.get(track_id, 0)
                        track_info['stamina'] = self.player_stamina.get(track_id, 100)
                        track_info['jump_count'] = int(self.player_jump_counts.get(track_id, 0))
                        track_info['status'] = self.player_status.get(track_id, 'waiting')

        self._apply_default_stats_to_tracks(tracks)

    def _apply_default_stats_to_tracks(self, tracks):
        """Ensure every tracked player has complete stat fields in every visible frame."""
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue

            for frame_players in object_tracks:
                for track_id, track_info in frame_players.items():
                    self._ensure_player_initialized(track_id)
                    track_info.setdefault('speed', 0.0)
                    track_info.setdefault('distance', float(self.player_total_distances.get(track_id, 0.0)))
                    track_info.setdefault('sprint_speed', float(self.player_sprint_speeds.get(track_id, 0.0)))
                    track_info.setdefault('acceleration', float(self.player_current_accelerations.get(track_id, 0.0)))
                    track_info.setdefault('max_acceleration', float(self.player_accelerations.get(track_id, 0.0)))
                    track_info.setdefault('stamina', float(self.player_stamina.get(track_id, 100.0)))
                    track_info.setdefault('jump_count', int(self.player_jump_counts.get(track_id, 0)))
                    track_info.setdefault('status', self.player_status.get(track_id, 'waiting'))

    def _update_player_stats(self, track_id, current_speed, distance_covered, start_pos, end_pos, frame_num):
        """Update enhanced player statistics"""
        # Initialize player data if not exists
        self._ensure_player_initialized(track_id)
        
        # Update speed history
        self.player_speeds_history[track_id].append(current_speed)
        if len(self.player_speeds_history[track_id]) > 10:  # Keep last 10 speeds
            self.player_speeds_history[track_id].pop(0)
        
        # Update position history
        self.player_positions_history[track_id].append((start_pos, end_pos, frame_num))
        if len(self.player_positions_history[track_id]) > 5:
            self.player_positions_history[track_id].pop(0)
        
        # Calculate current acceleration (change in speed over time)
        if len(self.player_speeds_history[track_id]) >= 2:
            # Get the last two speeds for current acceleration
            current_speed_ms = current_speed / 3.6  # Convert km/h to m/s
            prev_speed_ms = self.player_speeds_history[track_id][-2] / 3.6  # Convert km/h to m/s
            
            # Time difference between measurements
            time_diff = self.frame_window / self.frame_rate
            
            # Calculate current acceleration (m/s²)
            current_acceleration = (current_speed_ms - prev_speed_ms) / time_diff if time_diff > 0 else 0
            
            # Store current acceleration (can be positive or negative)
            self.player_current_accelerations[track_id] = current_acceleration
            
            # Update maximum absolute acceleration for reference
            self.player_accelerations[track_id] = max(self.player_accelerations[track_id], abs(current_acceleration))
        
        # Update sprint speed (maximum speed)
        self.player_sprint_speeds[track_id] = max(self.player_sprint_speeds[track_id], current_speed)
        
        # Update total distance
        self.player_total_distances[track_id] += distance_covered
        
        # Update player status based on current speed
        self._update_player_status(track_id, current_speed)
        
        # Update stamina (decreases with high speed activity)
        if current_speed > 20:  # High speed threshold
            stamina_decrease = (current_speed / 50) * 0.5  # Stamina decrease rate
            self.player_stamina[track_id] = max(0, self.player_stamina[track_id] - stamina_decrease)
        else:
            # Recover stamina slowly when not sprinting
            self.player_stamina[track_id] = min(100, self.player_stamina[track_id] + 0.1)
    
    def _update_player_status(self, track_id, current_speed):
        """Determine player status based on current speed"""
        if current_speed < self.status_thresholds['waiting']:
            status = 'waiting'
        elif current_speed < self.status_thresholds['jogging']:
            status = 'jogging'
        elif current_speed < self.status_thresholds['running']:
            status = 'running'
        else:
            status = 'sprinting'
        
        self.player_status[track_id] = status
    
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            # Store current frame number for jump storage reference
            self._current_frame_num = frame_num
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for track_id, track_info in object_tracks[frame_num].items():
                   # Always show stats for all tracked players, using class storage when track data is missing
                   speed = track_info.get('speed', 0)
                   distance = track_info.get('distance', self.player_total_distances.get(track_id, 0))
                   
                   # Get enhanced stats from class storage (fallback to current speeds if available)
                   if track_id in self.player_speeds_history and self.player_speeds_history[track_id]:
                       if speed == 0:  # If no speed in track_info, use latest from history
                           speed = self.player_speeds_history[track_id][-1]
                   
                   sprint_speed = self.player_sprint_speeds.get(track_id, speed)
                   acceleration = self.player_current_accelerations.get(track_id, 0)
                   stamina = self.player_stamina.get(track_id, 100)
                   jump_count = int(track_info.get('jump_count', self.player_jump_counts.get(track_id, 0)))
                   status = self.player_status.get(track_id, 'waiting')
                   
                   display_jump_count = jump_count
                   
                   bbox = track_info['bbox']
                   position = get_foot_position(bbox)
                   position = list(position)
                   position[1] += 40

                   # Create enhanced real-time interface (grey background panels)
                   self._draw_enhanced_player_stats(frame, position, track_id, {
                       'speed': speed,
                       'distance': distance,
                       'sprint_speed': sprint_speed,
                       'acceleration': acceleration,
                       'stamina': stamina,
                       'jump_count': display_jump_count,
                       'status': status,
                       })
            output_frames.append(frame)
        
        return output_frames

    def _draw_enhanced_player_stats(self, frame, position, player_id, stats):
        """Draw enhanced real-time player statistics interface"""
        try:
            x, y = int(position[0]), int(position[1])
            
            # Background panel for better readability (semi-transparent)
            panel_width = 200
            panel_height = 140  # Increased height for status
            panel_x = max(0, min(x - panel_width//2, frame.shape[1] - panel_width))
            panel_y = max(0, min(y, frame.shape[0] - panel_height))
            
            # Create semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (255, 255, 255), 2)
            
            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            line_height = 15
            
            # Player ID header
            text_y = panel_y + 15
            cv2.putText(frame, f"Player {player_id}", 
                       (panel_x + 5, text_y), font, 0.5, (255, 255, 255), 2)
            
            # Current Speed
            text_y += line_height
            speed_color = (0, 255, 0) if stats['speed'] < 15 else (0, 255, 255) if stats['speed'] < 25 else (0, 0, 255)
            cv2.putText(frame, f"Speed: {stats['speed']:.1f} km/h", 
                       (panel_x + 5, text_y), font, font_scale, speed_color, thickness)
            
            # Sprint Speed (Max Speed)
            text_y += line_height
            cv2.putText(frame, f"Sprint: {stats['sprint_speed']:.1f} km/h", 
                       (panel_x + 5, text_y), font, font_scale, (255, 100, 0), thickness)
            
            # Acceleration (show with sign to indicate acceleration/deceleration)
            text_y += line_height
            accel_value = stats['acceleration']
            if accel_value > 2:  # Strong acceleration
                accel_color = (0, 0, 255)  # Red for high acceleration
            elif accel_value > 0.5:  # Moderate acceleration
                accel_color = (0, 255, 255)  # Yellow for moderate acceleration  
            elif accel_value > -0.5:  # Steady speed
                accel_color = (0, 255, 0)  # Green for steady
            elif accel_value > -2:  # Moderate deceleration
                accel_color = (255, 255, 0)  # Yellow for moderate deceleration
            else:  # Strong deceleration
                accel_color = (255, 0, 0)  # Blue for strong deceleration
            
            # Show acceleration with + or - sign
            accel_sign = "+" if accel_value >= 0 else ""
            cv2.putText(frame, f"Accel: {accel_sign}{accel_value:.1f} m/s²", 
                       (panel_x + 5, text_y), font, font_scale, accel_color, thickness)
        
            # Distance
            text_y += line_height
            cv2.putText(frame, f"Dist: {stats['distance']:.0f} m", 
                       (panel_x + 5, text_y), font, font_scale, (255, 255, 0), thickness)
            
            # Stamina with progress bar
            text_y += line_height
            stamina_color = (0, 255, 0) if stats['stamina'] > 70 else (0, 255, 255) if stats['stamina'] > 30 else (0, 0, 255)
            cv2.putText(frame, f"Stamina: {stats['stamina']:.0f}%", 
                       (panel_x + 5, text_y), font, font_scale, stamina_color, thickness)
            
            # Stamina bar
            bar_x = panel_x + 5
            bar_y = text_y + 5
            bar_width = int((panel_width - 10) * (stats['stamina'] / 100))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 5), stamina_color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (panel_x + panel_width - 5, bar_y + 5), (128, 128, 128), 1)
            
            # Jump Count
            text_y += 15
            actual_jump_count = int(stats.get('jump_count', 0))
            jump_text = f"Jumps: {actual_jump_count}"
            cv2.putText(frame, jump_text, 
                       (panel_x + 5, text_y), font, font_scale, (255, 0, 255), thickness)

            
        except Exception as e:
            print(f"❌ Error drawing stats for player {player_id}: {e}")
            import traceback
            traceback.print_exc()

    def save_enhanced_stats_to_json(self, output_dir="output_data"):
        """Save enhanced player statistics to JSON file"""
        import json
        import os
        from datetime import datetime
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/enhanced_player_stats_{timestamp}.json"
        
        stats_data = self.build_enhanced_stats_data()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        return filename

    def build_enhanced_stats_data(self, all_player_ids=None):
        """Build enhanced player stats dictionary for export/aggregation."""
        if all_player_ids is None:
            all_player_ids = set(self.all_detected_player_ids)
            all_player_ids.update(self.player_speeds_history.keys())
            all_player_ids.update(self.player_total_distances.keys())

        stats_data = {}
        for player_id in sorted(all_player_ids):
            speed_history = self.player_speeds_history.get(player_id, [])
            stats_data[f"player_{int(player_id)}"] = {
                "player_id": int(player_id),
                "max_speed_kmh": float(self.player_sprint_speeds.get(player_id, 0)),
                "avg_speed_kmh": float(sum(speed_history) / len(speed_history) if speed_history else 0),
                "max_acceleration": float(self.player_accelerations.get(player_id, 0)),
                "current_acceleration": float(self.player_current_accelerations.get(player_id, 0)),
                "total_distance_m": float(self.player_total_distances.get(player_id, 0)),
                "jump_count": int(self.player_jump_counts.get(player_id, 0)),
                "final_stamina_percentage": float(self.player_stamina.get(player_id, 100)),
                "speed_history": [float(s) for s in speed_history],
                "jumps_detected": []
            }

        return stats_data