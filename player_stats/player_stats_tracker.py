import cv2
import numpy as np
import json
import os
from datetime import datetime


class PlayerStatsTracker:
    def __init__(self, frame_rate=24):
        self.frame_rate = frame_rate
        self.player_stats = {}
        self.frame_time = 1.0 / frame_rate  # Time per frame in seconds
        
    def update_player_stats(self, tracks):
        """Update statistics for all tracked players"""
        
        for frame_num in range(len(tracks['players'])):
            player_tracks = tracks['players'][frame_num]
            
            for player_id, track_info in player_tracks.items():
                if player_id not in self.player_stats:
                    # Initialize new player stats
                    self.player_stats[player_id] = {
                        'positions': [],
                        'speeds': [],
                        'accelerations': [],
                        'jump_count': 0,
                        'total_distance': 0.0,
                        'max_speed': 0.0,
                        'sprint_speed': 0.0,
                        'current_speed': 0.0,
                        'acceleration': 0.0,
                        'stamina_score': 100.0,
                        'sprint_time': 0.0,
                        'frame_count': 0,
                        'avg_speed': 0.0
                    }
                
                # Update current frame stats
                stats = self.player_stats[player_id]
                
                # Get position and speed from tracks
                if 'position_adjusted' in track_info:
                    position = track_info['position_adjusted']
                    stats['positions'].append(position)
                    
                    # Calculate current speed
                    if 'speed' in track_info:
                        current_speed = track_info['speed'] * 3.6  # Convert m/s to km/h
                        stats['speeds'].append(current_speed)
                        stats['current_speed'] = current_speed
                        stats['max_speed'] = max(stats['max_speed'], current_speed)
                        
                        # Update sprint speed (speeds > 20 km/h)
                        if current_speed > 20:
                            stats['sprint_speed'] = max(stats['sprint_speed'], current_speed)
                            stats['sprint_time'] += self.frame_time
                    
                    # Calculate distance
                    if 'distance' in track_info:
                        stats['total_distance'] += track_info['distance']
                
                # Calculate acceleration
                if len(stats['speeds']) >= 2:
                    speed_change = stats['speeds'][-1] - stats['speeds'][-2]
                    acceleration = abs(speed_change / self.frame_time)  # km/h per second
                    stats['accelerations'].append(acceleration)
                    stats['acceleration'] = acceleration
                
                # Detect jumps (significant vertical movement)
                if len(stats['positions']) >= 2:
                    prev_pos = stats['positions'][-2]
                    curr_pos = stats['positions'][-1]
                    vertical_change = abs(curr_pos[1] - prev_pos[1])
                    
                    # Jump detection threshold
                    if vertical_change > 15:  # Pixels - adjust as needed
                        stats['jump_count'] += 1
                
                # Calculate stamina (decreases with high activity)
                if stats['current_speed'] > 15:
                    stamina_loss = 0.1  # Lose stamina when running fast
                    stats['stamina_score'] = max(0, stats['stamina_score'] - stamina_loss)
                elif stats['current_speed'] < 5:
                    stamina_recovery = 0.05  # Recover stamina when resting
                    stats['stamina_score'] = min(100, stats['stamina_score'] + stamina_recovery)
                
                # Update frame count and average speed
                stats['frame_count'] += 1
                if stats['speeds']:
                    stats['avg_speed'] = sum(stats['speeds']) / len(stats['speeds'])
    
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
            f"Accel: {stats['acceleration']:.0f}",       # Acceleration
            f"Jumps: {stats['jump_count']}",             # Jump Count
            f"Stamina: {stats['stamina_score']:.0f}%",   # Stamina
            f"Dist: {stats['total_distance']:.0f}m"      # Running Distance
        ]
        
        # Color coding for different stats
        colors = [
            (255, 255, 255),  # White for Player ID
            (0, 255, 255) if stats['current_speed'] > 15 else (255, 255, 255),  # Cyan for high speed
            (0, 255, 0) if stats['sprint_speed'] > 20 else (255, 255, 255),     # Green for sprint
            (255, 128, 0) if stats['acceleration'] > 100 else (255, 255, 255),  # Orange for high accel
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
    
    def save_stats_to_file(self, output_dir="output_data"):
        """Save player statistics to JSON file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"player_stats_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data for JSON serialization
        stats_data = {}
        for player_id, stats in self.player_stats.items():
            # Convert all types to standard Python types for JSON serialization
            clean_stats = {
                'player_id': int(player_id),  # Convert numpy int64 to int
                'total_distance_m': float(stats['total_distance']),
                'max_speed_kmh': float(stats['max_speed']),
                'avg_speed_kmh': float(stats['avg_speed']),
                'sprint_speed_kmh': float(stats['sprint_speed']),
                'max_acceleration': float(max(stats['accelerations']) if stats['accelerations'] else 0),
                'jump_count': int(stats['jump_count']),
                'stamina_percentage': float(stats['stamina_score']),
                'sprint_time_seconds': float(stats['sprint_time']),
                'total_sprint_distance_m': float(stats['sprint_speed'] * stats['sprint_time'] / 3.6) if stats['sprint_speed'] > 0 else 0.0,
                'total_frames_tracked': int(stats['frame_count'])
            }
            stats_data[f'player_{int(player_id)}'] = clean_stats
        
        # Add summary statistics
        stats_data['match_summary'] = {
            'total_players': len(self.player_stats),
            'timestamp': timestamp,
            'analysis_duration_frames': max([stats['frame_count'] for stats in self.player_stats.values()]) if self.player_stats else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        return filepath