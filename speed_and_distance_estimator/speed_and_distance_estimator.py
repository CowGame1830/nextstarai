import cv2
import sys 
import numpy as np
import sys
sys.path.append('../')
from utils import measure_distance, get_foot_position
from pose_estimator.advanced_jump_detector_clean import AdvancedJumpDetector
import cv2
import numpy as np
import json
import os

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
        # Enhanced tracking for real-time stats
        self.player_positions_history = {}
        self.player_speeds_history = {}
        self.player_accelerations = {}  # Maximum acceleration
        self.player_current_accelerations = {}  # Current acceleration
        self.player_jump_detection = {}
        self.player_stamina = {}
        self.player_sprint_speeds = {}
        self.player_total_distances = {}
        self.player_status = {}  # Current player status (waiting/jogging/running/sprinting)
        
        # Speed thresholds for status determination (km/h)
        self.status_thresholds = {
            'waiting': 3.0,      # 0-3 km/h
            'jogging': 12.0,     # 3-12 km/h  
            'running': 20.0,     # 12-20 km/h
            'sprinting': float('inf')  # 20+ km/h
        }
        
        # Initialize advanced jump detector
        self.jump_detector = AdvancedJumpDetector()
        self.video_frames = []  # Store frames for jump detection processing
    
    def set_video_frames(self, frames):
        """Store video frames for advanced jump detection processing"""
        self.video_frames = frames
    
    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    # Find the next frame where this player appears instead of strict last_frame
                    actual_end_frame = None
                    for check_frame in range(last_frame, frame_num, -1):  # Search backwards from last_frame
                        if track_id in object_tracks[check_frame]:
                            actual_end_frame = check_frame
                            break
                    
                    # If player not found in any frame in the window, skip
                    if actual_end_frame is None or actual_end_frame == frame_num:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[actual_end_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (actual_end_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    # Enhanced tracking for real-time stats
                    self._update_player_stats(track_id, speed_km_per_hour, distance_covered, 
                                            start_position, end_position, frame_num)

                    for frame_num_batch in range(frame_num,actual_end_frame+1):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
                        # Add enhanced stats to tracks
                        tracks[object][frame_num_batch][track_id]['sprint_speed'] = self.player_sprint_speeds.get(track_id, 0)
                        tracks[object][frame_num_batch][track_id]['acceleration'] = self.player_current_accelerations.get(track_id, 0)  # Use current acceleration
                        tracks[object][frame_num_batch][track_id]['max_acceleration'] = self.player_accelerations.get(track_id, 0)  # Keep max for reference
                        tracks[object][frame_num_batch][track_id]['stamina'] = self.player_stamina.get(track_id, 100)
                        tracks[object][frame_num_batch][track_id]['jump_count'] = len(self.player_jump_detection.get(track_id, []))
                        tracks[object][frame_num_batch][track_id]['status'] = self.player_status.get(track_id, 'waiting')

    def _update_player_stats(self, track_id, current_speed, distance_covered, start_pos, end_pos, frame_num):
        """Update enhanced player statistics"""
        # Initialize player data if not exists
        if track_id not in self.player_speeds_history:
            self.player_speeds_history[track_id] = []
            self.player_accelerations[track_id] = 0  # Maximum acceleration
            self.player_current_accelerations[track_id] = 0  # Current acceleration
            self.player_jump_detection[track_id] = []
            self.player_stamina[track_id] = 100
            self.player_sprint_speeds[track_id] = 0
            self.player_total_distances[track_id] = 0
            self.player_positions_history[track_id] = []
            self.player_status[track_id] = 'waiting'
        
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
        
        # Advanced jump detection (accurate OpenCV-based)
        if self.video_frames and frame_num < len(self.video_frames):
            try:
                # Get current frame player tracks for advanced jump detection
                current_frame_players = {track_id: {'bbox': [start_pos[0], start_pos[1], end_pos[0], end_pos[1]]}}
                
                # Process jump detection with advanced detector
                _, jump_data = self.jump_detector.process_frame(
                    self.video_frames[frame_num], 
                    current_frame_players, 
                    frame_num
                )
                
                # Update jump detection data
                if track_id in jump_data:
                    jump_info = jump_data[track_id]
                    if jump_info.get('is_jumping', False) and jump_info.get('jump_phase') == 'takeoff':
                        jump_height = jump_info.get('jump_height', 0)
                        # Only add jump when takeoff phase is detected (prevents duplicate counting)
                        if frame_num not in [jump[0] for jump in self.player_jump_detection[track_id]]:
                            self.player_jump_detection[track_id].append((frame_num, jump_height))
                            print(f"[JUMP STORED] Player {track_id}: Jump {len(self.player_jump_detection[track_id])} stored at frame {frame_num}, height: {jump_height:.1f}px")
                            
            except Exception as e:
                # Fallback to basic jump detection if advanced detector fails
                if len(self.player_positions_history[track_id]) >= 2:
                    prev_pos = self.player_positions_history[track_id][-2][1]  # Previous end position
                    curr_pos = end_pos
                    if prev_pos and curr_pos:
                        vertical_change = abs(prev_pos[1] - curr_pos[1])
                        if vertical_change > 30:  # Fallback jump threshold
                            if frame_num not in [jump[0] for jump in self.player_jump_detection[track_id]]:
                                self.player_jump_detection[track_id].append((frame_num, vertical_change))
                                print(f"[FALLBACK JUMP] Player {track_id}: Jump stored at frame {frame_num}, vertical change: {vertical_change:.1f}px")
        
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
                   jump_count = len(self.player_jump_detection.get(track_id, []))
                   status = self.player_status.get(track_id, 'waiting')
                   
                   # Check if jump is currently detected in this frame
                   is_jumping_now = track_info.get('advanced_jump', False)
                   jump_phase = track_info.get('jump_phase', 'unknown')
                   
                   # Update jump count in real-time when jump is detected
                   current_jump_count = jump_count
                   
                   # Store jump in real-time if currently jumping and not already stored
                   if is_jumping_now and jump_phase == 'takeoff':
                       # Check if this frame's jump is not already stored
                       existing_jumps = [jump[0] for jump in self.player_jump_detection.get(track_id, [])]
                       if frame_num not in existing_jumps:
                           # Store the jump immediately
                           jump_height = track_info.get('jump_height_advanced', 20)  # Default height if not available
                           if track_id not in self.player_jump_detection:
                               self.player_jump_detection[track_id] = []
                           self.player_jump_detection[track_id].append((frame_num, jump_height))
                           print(f"[REAL-TIME JUMP] Player {track_id}: Jump {len(self.player_jump_detection[track_id])} stored at frame {frame_num}")
                           
                           # Update the tracks data immediately
                           tracks[object][frame_num][track_id]['jump_count'] = len(self.player_jump_detection[track_id])
                           current_jump_count = len(self.player_jump_detection[track_id])
                   
                   # Show +1 during takeoff and airborne phases
                   if is_jumping_now and jump_phase in ['takeoff', 'airborne']:
                       # Display the incremented count during jump phases
                       display_jump_count = max(current_jump_count, jump_count + 1)
                   else:
                       display_jump_count = current_jump_count
                   
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
                           'is_jumping_now': is_jumping_now,
                           'jump_phase': jump_phase,
                           'jump_height_advanced': track_info.get('jump_height_advanced', 0)
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
            
            # Jump Count with real-time "+1" indicator - always show stored jump count
            text_y += 15
            # Ensure we always show the actual stored jump count
            if player_id not in self.player_jump_detection:
                self.player_jump_detection[player_id] = []
            actual_jump_count = len(self.player_jump_detection[player_id])
            jump_text = f"Jumps: {actual_jump_count}"
            cv2.putText(frame, jump_text, 
                       (panel_x + 5, text_y), font, font_scale, (255, 0, 255), thickness)
            
            # Show "+1" indicator when player is currently jumping and store the jump
            if stats.get('is_jumping_now', False) and stats.get('jump_phase') in ['takeoff', 'airborne']:
                # Initialize player jump detection if not exists
                if player_id not in self.player_jump_detection:
                    self.player_jump_detection[player_id] = []
                
                # Store jump when +1 is displayed (during takeoff phase only to prevent duplicates)
                if stats.get('jump_phase') == 'takeoff':
                    # Check if jump for current frame is not already stored
                    frame_num = getattr(self, '_current_frame_num', 0)  # Get current frame number
                    existing_jumps = [jump[0] for jump in self.player_jump_detection[player_id]]
                    if frame_num not in existing_jumps:
                        # Store the jump: increment count from current to +1
                        jump_height = stats.get('jump_height_advanced', 25.0)  # Use advanced jump height if available
                        self.player_jump_detection[player_id].append((frame_num, jump_height))
                        new_count = len(self.player_jump_detection[player_id])
                        print(f"[REAL-TIME JUMP] Player {player_id}: Jump {new_count} stored at frame {frame_num}")
                        
                        # Update the stats dictionary to reflect the new count immediately
                        stats['jump_count'] = new_count
                
                # Position "+1" indicator next to jump count within the grey panel
                jump_indicator_x = panel_x + 120  # Position within the panel
                jump_indicator_y = text_y        # Same line as jump count
                
                # Draw prominent "+1" indicator with outline for visibility in grey background
                cv2.putText(frame, "+1", 
                           (jump_indicator_x, jump_indicator_y), font, 0.7, (0, 0, 0), 3)  # Black outline
                cv2.putText(frame, "+1", 
                           (jump_indicator_x, jump_indicator_y), font, 0.7, (0, 255, 0), 2)  # Bright green "+1"

            
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
        
        stats_data = {}
        for player_id in self.player_speeds_history.keys():
            stats_data[f"player_{int(player_id)}"] = {
                "player_id": int(player_id),
                "max_speed_kmh": float(self.player_sprint_speeds.get(player_id, 0)),
                "avg_speed_kmh": float(np.mean(self.player_speeds_history.get(player_id, [0])) if self.player_speeds_history.get(player_id) else 0),
                "max_acceleration": float(self.player_accelerations.get(player_id, 0)),
                "current_acceleration": float(self.player_current_accelerations.get(player_id, 0)),
                "total_distance_m": float(self.player_total_distances.get(player_id, 0)),
                "jump_count": len(self.player_jump_detection.get(player_id, [])),
                "final_stamina_percentage": float(self.player_stamina.get(player_id, 100)),
                "speed_history": [float(s) for s in self.player_speeds_history.get(player_id, [])],
                "jumps_detected": [(int(frame), float(height)) for frame, height in self.player_jump_detection.get(player_id, [])]
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        return filename