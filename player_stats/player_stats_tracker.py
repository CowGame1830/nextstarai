import cv2
import numpy as np
import json
import os
from datetime import datetime
from collections import deque


class PlayerStatsTracker:
    def __init__(self, frame_rate=24):
        self.frame_rate = frame_rate
        self.player_stats = {}
        self.frame_time = 1.0 / frame_rate  # Time per frame in seconds
        self.jump_cooldown_frames = max(4, int(0.35 * frame_rate))
        self.max_plausible_speed_kmh = 42.0
        self.accel_smoothing_alpha = 0.45
        self.accel_deadband_ms2 = 0.25
        self.max_plausible_acceleration_ms2 = 6.5
        
    def update_player_stats(self, tracks):
        """Update statistics for all tracked players"""
        
        for frame_num in range(len(tracks['players'])):
            player_tracks = tracks['players'][frame_num]
            
            for player_id, track_info in player_tracks.items():
                if player_id not in self.player_stats:
                    # Initialize new player stats
                    self.player_stats[player_id] = {
                        'positions': deque(maxlen=2),
                        'jump_count': 0,
                        'total_distance': 0.0,
                        'max_speed': 0.0,
                        'sprint_speed': 0.0,
                        'current_speed': 0.0,
                        'acceleration': 0.0,
                        'max_acceleration': 0.0,
                        'speed_sum': 0.0,
                        'speed_count': 0,
                        'stamina_score': 100.0,
                        'sprint_time': 0.0,
                        'sprint_distance': 0.0,
                        'frame_count': 0,
                        'avg_speed': 0.0,
                        'last_jump_frame': -10_000,
                        'last_speed_for_accel': None,
                        'last_speed_frame': None,
                        'jump_state': 'ground',
                        'jump_up_frames': 0,
                        'jump_down_frames': 0,
                        'jump_peak_rise': 0.0,
                        'jump_takeoff_frame': -10_000,
                    }
                
                # Update current frame stats
                stats = self.player_stats[player_id]
                
                # Get position and speed from tracks
                if 'position_adjusted' in track_info:
                    position = track_info['position_adjusted']
                    stats['positions'].append(position)
                    
                    # Calculate current speed
                    if 'speed' in track_info:
                        # track_info['speed'] is already in km/h from SpeedAndDistance_Estimator
                        current_speed = float(track_info['speed'])
                        current_speed = float(np.clip(current_speed, 0.0, self.max_plausible_speed_kmh))
                        stats['current_speed'] = current_speed
                        stats['max_speed'] = max(stats['max_speed'], current_speed)
                        stats['speed_sum'] += current_speed
                        stats['speed_count'] += 1
                        stats['avg_speed'] = stats['speed_sum'] / stats['speed_count']
                        
                        # Update sprint speed (speeds > 20 km/h)
                        if current_speed > 20:
                            stats['sprint_speed'] = max(stats['sprint_speed'], current_speed)
                            stats['sprint_time'] += self.frame_time
                            stats['sprint_distance'] += (current_speed / 3.6) * self.frame_time

                        # Use actual frame delta for acceleration so duplicated window speeds don't explode values.
                        if stats['last_speed_for_accel'] is not None and stats['last_speed_frame'] is not None:
                            frame_delta = max(1, frame_num - stats['last_speed_frame'])
                            dt = frame_delta * self.frame_time
                            prev_speed_ms = stats['last_speed_for_accel'] / 3.6
                            curr_speed_ms = current_speed / 3.6
                            raw_acceleration = (curr_speed_ms - prev_speed_ms) / dt
                            if abs(raw_acceleration) < self.accel_deadband_ms2:
                                raw_acceleration = 0.0
                            raw_acceleration = float(np.clip(
                                raw_acceleration,
                                -self.max_plausible_acceleration_ms2,
                                self.max_plausible_acceleration_ms2,
                            ))

                            prev_acc = float(stats.get('acceleration', 0.0))
                            acceleration = (
                                self.accel_smoothing_alpha * raw_acceleration
                                + (1.0 - self.accel_smoothing_alpha) * prev_acc
                            )
                            stats['acceleration'] = acceleration
                            stats['max_acceleration'] = max(stats['max_acceleration'], abs(acceleration))

                        stats['last_speed_for_accel'] = current_speed
                        stats['last_speed_frame'] = frame_num
                    
                    # Calculate distance
                    if 'distance' in track_info:
                        # track_info['distance'] is cumulative for the player.
                        incoming_distance = float(track_info['distance'])
                        if np.isfinite(incoming_distance):
                            stats['total_distance'] = max(stats['total_distance'], max(0.0, incoming_distance))
                
                # Detect jumps with a rise->fall pattern to avoid one-frame jitter counts.
                if len(stats['positions']) >= 2:
                    prev_pos = stats['positions'][-2]
                    curr_pos = stats['positions'][-1]
                    vertical_delta = curr_pos[1] - prev_pos[1]

                    bbox = track_info.get('bbox', [0, 0, 0, 0])
                    bbox_height = max(1.0, float(bbox[3] - bbox[1]))
                    up_threshold = max(6.0, 0.08 * bbox_height)
                    down_threshold = max(4.0, 0.05 * bbox_height)
                    max_jump_frames = max(10, int(0.9 * self.frame_rate))

                    state = stats['jump_state']
                    can_start_jump = (frame_num - stats['last_jump_frame']) >= self.jump_cooldown_frames

                    if state == 'ground':
                        if vertical_delta < -up_threshold and can_start_jump:
                            stats['jump_state'] = 'rising'
                            stats['jump_up_frames'] = 1
                            stats['jump_down_frames'] = 0
                            stats['jump_peak_rise'] = -vertical_delta
                            stats['jump_takeoff_frame'] = frame_num

                    elif state == 'rising':
                        if vertical_delta < -(0.6 * up_threshold):
                            stats['jump_up_frames'] += 1
                            stats['jump_peak_rise'] += -vertical_delta
                        elif vertical_delta > down_threshold:
                            stats['jump_state'] = 'falling'
                            stats['jump_down_frames'] = 1

                        if (frame_num - stats['jump_takeoff_frame']) > max_jump_frames:
                            stats['jump_state'] = 'ground'
                            stats['jump_up_frames'] = 0
                            stats['jump_down_frames'] = 0
                            stats['jump_peak_rise'] = 0.0

                    elif state == 'falling':
                        if vertical_delta > (0.6 * down_threshold):
                            stats['jump_down_frames'] += 1

                        if stats['jump_down_frames'] >= 2:
                            min_peak_rise = max(12.0, 0.18 * bbox_height)
                            if stats['jump_up_frames'] >= 2 and stats['jump_peak_rise'] >= min_peak_rise:
                                stats['jump_count'] += 1
                                stats['last_jump_frame'] = frame_num

                            stats['jump_state'] = 'ground'
                            stats['jump_up_frames'] = 0
                            stats['jump_down_frames'] = 0
                            stats['jump_peak_rise'] = 0.0
                
                # Calculate stamina (decreases with high activity)
                if stats['current_speed'] > 15:
                    stamina_loss = 0.1  # Lose stamina when running fast
                    stats['stamina_score'] = max(0, stats['stamina_score'] - stamina_loss)
                elif stats['current_speed'] < 5:
                    stamina_recovery = 0.05  # Recover stamina when resting
                    stats['stamina_score'] = min(100, stats['stamina_score'] + stamina_recovery)
                
                # Update frame count
                stats['frame_count'] += 1
    
    def get_stamina_color(self, stamina):
        """Get color based on stamina level"""
        if stamina > 70:
            return (0, 255, 0)      # Green - Good stamina
        elif stamina > 40:
            return (0, 255, 255)    # Yellow - Medium stamina
        else:
            return (0, 0, 255)      # Red - Low stamina
    
    def draw_realtime_player_interface(self, frames, tracks):
        """Draw real-time stats interface below each player without background"""
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            # Draw stats for each player in current frame
            if frame_num < len(tracks['players']):
                player_tracks = tracks['players'][frame_num]
                
                for player_id, track_info in player_tracks.items():
                    if player_id in self.player_stats:
                        bbox = track_info['bbox']
                        stats = self.player_stats[player_id]
                        
                        # Draw real-time stats below each player (no background)
                        frame = self.draw_player_stats_below(frame, player_id, bbox, stats)
            
            output_frames.append(frame)
        
        return output_frames
    
    def draw_player_stats_below(self, frame, player_id, bbox, stats):
        """Draw player statistics below each player without background"""
        if not stats:
            return frame
            
        # Position text below player
        center_x = int((bbox[0] + bbox[2]) / 2)
        stats_y = int(bbox[3]) + 20  # Below the player bbox
        
        # Ensure text stays within frame
        frame_height, frame_width = frame.shape[:2]
        stats_y = min(stats_y, frame_height - 120)  # Leave space for multiple lines
        
        # Font properties - clean and readable
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        line_height = 14
        
        # Real-time stats display (no background)
        stats_lines = [
            f"P{player_id}",  # Player ID
            f"Speed: {stats['current_speed']:.1f}km/h",  # Current Speed
            f"Sprint: {stats['sprint_speed']:.1f}km/h",  # Sprint Speed  
            f"Accel: {stats['acceleration']:+.1f}",      # Signed acceleration (m/s^2)
            f"Jumps: {stats['jump_count']}",             # Jump Count
            f"Stamina: {stats['stamina_score']:.0f}%",   # Stamina
            f"Dist: {stats['total_distance']:.0f}m"      # Running Distance
        ]
        
        # Color coding for different stats
        colors = [
            (255, 255, 255),  # White for Player ID
            (0, 255, 255) if stats['current_speed'] > 15 else (255, 255, 255),  # Cyan for high speed
            (0, 255, 0) if stats['sprint_speed'] > 20 else (255, 255, 255),     # Green for sprint
            (255, 128, 0) if abs(stats['acceleration']) > 3 else (255, 255, 255),  # Orange for high accel
            (255, 0, 255) if stats['jump_count'] > 0 else (255, 255, 255),      # Magenta for jumps
            self.get_stamina_color(stats['stamina_score']),                      # Color based on stamina
            (128, 255, 128)   # Light green for distance
        ]
        
        # Draw each stat line with outline for visibility (no background)
        for i, (text, color) in enumerate(zip(stats_lines, colors)):
            text_x = center_x - 45  # Center the text
            text_y = stats_y + (i * line_height)
            
            # Draw black outline for better visibility
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1)
            # Draw main text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame
    
    def get_all_stats(self):
        """Get all player statistics"""
        return self.player_stats

    def build_stats_payload(self, timestamp=None):
        """Build player statistics payload dictionary for export/aggregation."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stats_data = {}
        for player_id, stats in self.player_stats.items():
            stamina_percentage = float(stats['stamina_score'])
            clean_stats = {
                'player_id': int(player_id),
                'total_distance_m': float(stats['total_distance']),
                'max_speed_kmh': float(stats['max_speed']),
                'avg_speed_kmh': float(stats['avg_speed']),
                'sprint_speed_kmh': float(stats['sprint_speed']),
                'max_acceleration': float(stats['max_acceleration']),
                'jump_count': int(stats['jump_count']),
                'stamina_percentage': stamina_percentage,
                'stamina_diff': float(100.0 - stamina_percentage),
                'sprint_time_seconds': float(stats['sprint_time']),
                'total_sprint_distance_m': float(min(stats['total_distance'], stats['sprint_distance'])),
                'total_frames_tracked': int(stats['frame_count'])
            }
            stats_data[f'player_{int(player_id)}'] = clean_stats

        stats_data['match_summary'] = {
            'total_players': len(self.player_stats),
            'timestamp': timestamp,
            'analysis_duration_frames': max([stats['frame_count'] for stats in self.player_stats.values()]) if self.player_stats else 0
        }

        return stats_data
    
    def save_stats_to_file(self, output_dir="output_data", filename_prefix="player_stats", timestamp=None):
        """Save player statistics to JSON file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename with timestamp
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        stats_data = self.build_stats_payload(timestamp=timestamp)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2)
        
        return filepath

    def save_player_stats_per_file(self, output_dir="output_data", filename_prefix="player", timestamp=None):
        """Save one JSON file per player and return file paths."""
        player_dir = os.path.join(output_dir, "players")
        if not os.path.exists(player_dir):
            os.makedirs(player_dir)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_paths = []

        for player_id, stats in self.player_stats.items():
            stamina_percentage = float(stats['stamina_score'])
            player_payload = {
                'player_id': int(player_id),
                'timestamp': timestamp,
                'total_distance_m': float(stats['total_distance']),
                'max_speed_kmh': float(stats['max_speed']),
                'avg_speed_kmh': float(stats['avg_speed']),
                'sprint_speed_kmh': float(stats['sprint_speed']),
                'max_acceleration': float(stats['max_acceleration']),
                'jump_count': int(stats['jump_count']),
                'stamina_percentage': stamina_percentage,
                'stamina_diff': float(100.0 - stamina_percentage),
                'sprint_time_seconds': float(stats['sprint_time']),
                'total_sprint_distance_m': float(min(stats['total_distance'], stats['sprint_distance'])),
                'total_frames_tracked': int(stats['frame_count'])
            }

            filename = f"{filename_prefix}_{int(player_id)}_{timestamp}.json"
            filepath = os.path.join(player_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(player_payload, f, indent=2)

            file_paths.append(filepath)

        return file_paths