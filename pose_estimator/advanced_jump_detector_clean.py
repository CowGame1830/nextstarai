import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import os
from datetime import datetime


class AdvancedJumpDetector:
    """
    Advanced jump detection using OpenCV-based computer vision techniques
    This is an alternative to MediaPipe that works with any Python version
    """
    
    def __init__(self):
        """Initialize the advanced jump detector"""
        
        # Jump detection parameters
        self.jump_threshold = 15  # Minimum vertical displacement for jump (pixels)
        self.min_jump_duration = 2  # Minimum frames for a valid jump
        self.max_jump_duration = 15  # Maximum frames for a valid jump
        self.velocity_threshold = 8  # Minimum upward velocity for jump detection
        
        # Player tracking data
        self.player_positions_history = {}  # Store position history for each player
        self.player_velocities = {}         # Store velocity history
        self.player_jump_states = {}        # Current jump state for each player
        self.player_jump_counts = {}        # Total jumps for each player
        self.player_jump_heights = {}       # Maximum jump heights for each player
        self.player_ground_levels = {}      # Ground level estimation for each player
        self.player_bbox_heights = {}       # Track bbox height changes
        
        # Jump state definitions
        self.JUMP_STATES = {
            'GROUND': 0,
            'TAKEOFF': 1,
            'AIRBORNE': 2,
            'LANDING': 3
        }
        
        # Motion detection parameters
        self.history_window = 8  # Number of frames to analyze
        self.ground_estimation_window = 20  # Frames for ground level estimation
        
    def detect_jumps_from_tracking(self, player_tracks: Dict, frame_num: int, frame: np.ndarray) -> Dict:
        """
        Detect jumps using advanced tracking analysis
        
        Args:
            player_tracks: Dictionary of player tracks with bboxes
            frame_num: Current frame number
            frame: Current video frame
            
        Returns:
            Dictionary with jump information for each player
        """
        jump_data = {}
        
        for player_id, track_info in player_tracks.items():
            bbox = track_info['bbox']
            
            # Initialize player data if not exists
            if player_id not in self.player_positions_history:
                self._initialize_player_data(player_id)
            
            # Calculate key positions from bbox
            foot_position = self._get_foot_position_from_bbox(bbox)
            center_position = self._get_center_position_from_bbox(bbox)
            bbox_height = bbox[3] - bbox[1]  # Height of bounding box
            
            # Update position history
            self.player_positions_history[player_id].append({
                'frame': frame_num,
                'foot_y': foot_position[1],
                'center_x': center_position[0],
                'center_y': center_position[1],
                'bbox_height': bbox_height,
                'bbox': bbox
            })
            
            # Keep only recent history
            if len(self.player_positions_history[player_id]) > self.history_window:
                self.player_positions_history[player_id].pop(0)
            
            # Analyze jump for this player
            jump_info = self._analyze_jump_motion(player_id, frame_num, frame, bbox)
            jump_data[player_id] = jump_info
            
        return jump_data
    
    def _initialize_player_data(self, player_id: int):
        """Initialize tracking data for a new player"""
        self.player_positions_history[player_id] = []
        self.player_velocities[player_id] = []
        self.player_jump_states[player_id] = self.JUMP_STATES['GROUND']
        self.player_jump_counts[player_id] = 0
        self.player_jump_heights[player_id] = 0
        self.player_ground_levels[player_id] = []
        self.player_bbox_heights[player_id] = []
    
    def _get_foot_position_from_bbox(self, bbox: List[float]) -> Tuple[float, float]:
        """Get foot position (bottom center of bbox)"""
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2
        foot_y = y2  # Bottom of bbox
        return foot_x, foot_y
    
    def _get_center_position_from_bbox(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center position of bbox"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
    
    def _analyze_jump_motion(self, player_id: int, frame_num: int, frame: np.ndarray, bbox: List[float]) -> Dict:
        """
        Analyze jumping motion using multiple indicators
        
        Args:
            player_id: Player identifier
            frame_num: Current frame number
            frame: Current video frame
            bbox: Player bounding box
            
        Returns:
            Dictionary with jump analysis results
        """
        history = self.player_positions_history[player_id]
        
        if len(history) < 3:  # Need at least 3 frames for analysis
            return {'is_jumping': False, 'jump_height': 0, 'jump_phase': 'unknown'}
        
        # Calculate vertical velocity and acceleration
        current_pos = history[-1]
        prev_pos = history[-2] if len(history) >= 2 else current_pos
        prev_prev_pos = history[-3] if len(history) >= 3 else prev_pos
        
        # Vertical velocity (negative means upward movement)
        vertical_velocity = current_pos['foot_y'] - prev_pos['foot_y']
        
        # Vertical acceleration
        if len(history) >= 3:
            prev_velocity = prev_pos['foot_y'] - prev_prev_pos['foot_y']
            vertical_acceleration = vertical_velocity - prev_velocity
        else:
            vertical_acceleration = 0
        
        # Update ground level estimation
        self._update_ground_level(player_id, current_pos['foot_y'])
        
        # Get estimated ground level
        if self.player_ground_levels[player_id]:
            ground_level = np.percentile(self.player_ground_levels[player_id], 80)  # 80th percentile as ground
        else:
            ground_level = current_pos['foot_y']
        
        # Calculate height above ground
        height_above_ground = max(0, ground_level - current_pos['foot_y'])
        
        # Detect bbox compression/expansion (indicator of crouch/jump preparation)
        bbox_height_change = self._analyze_bbox_height_change(player_id, current_pos['bbox_height'])
        
        # Multi-factor jump detection
        jump_indicators = self._calculate_jump_indicators({
            'vertical_velocity': vertical_velocity,
            'vertical_acceleration': vertical_acceleration,
            'height_above_ground': height_above_ground,
            'bbox_height_change': bbox_height_change,
            'history_length': len(history)
        })
        
        # Update jump state machine
        jump_result = self._update_jump_state(player_id, jump_indicators, height_above_ground)
        
        return {
            'is_jumping': jump_result['is_jumping'],
            'jump_height': height_above_ground,
            'jump_phase': jump_result['phase'],
            'vertical_velocity': vertical_velocity,
            'vertical_acceleration': vertical_acceleration,
            'ground_level': ground_level,
            'bbox_height_change': bbox_height_change,
            'jump_confidence': jump_indicators['confidence']
        }
    
    def _update_ground_level(self, player_id: int, foot_y: float):
        """Update ground level estimation for a player"""
        self.player_ground_levels[player_id].append(foot_y)
        
        # Keep only recent measurements for ground estimation
        if len(self.player_ground_levels[player_id]) > self.ground_estimation_window:
            self.player_ground_levels[player_id].pop(0)
    
    def _analyze_bbox_height_change(self, player_id: int, current_height: float) -> float:
        """Analyze changes in bounding box height (crouching/jumping indicator)"""
        self.player_bbox_heights[player_id].append(current_height)
        
        if len(self.player_bbox_heights[player_id]) > 5:
            self.player_bbox_heights[player_id].pop(0)
        
        if len(self.player_bbox_heights[player_id]) >= 3:
            recent_avg = np.mean(self.player_bbox_heights[player_id][-3:])
            baseline_avg = np.mean(self.player_bbox_heights[player_id])
            height_change = (recent_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0
            return height_change
        
        return 0
    
    def _calculate_jump_indicators(self, motion_data: Dict) -> Dict:
        """Calculate jump indicators from motion analysis"""
        indicators = {
            'upward_motion': 0,
            'sudden_acceleration': 0,
            'height_threshold': 0,
            'bbox_compression': 0,
            'confidence': 0
        }
        
        # Upward motion indicator (negative velocity means upward)
        if motion_data['vertical_velocity'] < -self.velocity_threshold:
            indicators['upward_motion'] = min(1.0, abs(motion_data['vertical_velocity']) / 20)
        
        # Sudden upward acceleration
        if motion_data['vertical_acceleration'] < -5:  # Sudden upward acceleration
            indicators['sudden_acceleration'] = min(1.0, abs(motion_data['vertical_acceleration']) / 15)
        
        # Height above ground threshold
        if motion_data['height_above_ground'] > self.jump_threshold:
            indicators['height_threshold'] = min(1.0, motion_data['height_above_ground'] / 50)
        
        # Bbox compression (crouching before jump)
        if motion_data['bbox_height_change'] < -0.1:  # Bbox got smaller (crouch)
            indicators['bbox_compression'] = min(1.0, abs(motion_data['bbox_height_change']) * 5)
        
        # Calculate overall confidence
        weights = [0.4, 0.3, 0.2, 0.1]  # Weights for different indicators
        confidence = sum(w * v for w, v in zip(weights, indicators.values()))
        indicators['confidence'] = min(1.0, confidence)
        
        return indicators
    
    def _update_jump_state(self, player_id: int, indicators: Dict, height_above_ground: float) -> Dict:
        """Update jump state machine for a player"""
        current_state = self.player_jump_states[player_id]
        confidence = indicators['confidence']
        
        is_jumping = False
        phase = 'ground'
        
        if current_state == self.JUMP_STATES['GROUND']:
            # Look for takeoff: high confidence or significant upward motion
            if confidence > 0.6 or (indicators['upward_motion'] > 0.5 and indicators['sudden_acceleration'] > 0.3):
                self.player_jump_states[player_id] = self.JUMP_STATES['TAKEOFF']
                phase = 'takeoff'
                is_jumping = True
                
        elif current_state == self.JUMP_STATES['TAKEOFF']:
            # Confirm airborne: sustained elevation
            if height_above_ground > self.jump_threshold:
                self.player_jump_states[player_id] = self.JUMP_STATES['AIRBORNE']
                phase = 'airborne'
                is_jumping = True
                # Update max jump height
                self.player_jump_heights[player_id] = max(
                    self.player_jump_heights[player_id], 
                    height_above_ground
                )
            elif confidence < 0.3:
                # False alarm
                self.player_jump_states[player_id] = self.JUMP_STATES['GROUND']
                
        elif current_state == self.JUMP_STATES['AIRBORNE']:
            if height_above_ground > self.jump_threshold * 0.7:  # Still significantly elevated
                phase = 'airborne'
                is_jumping = True
                # Update max jump height
                self.player_jump_heights[player_id] = max(
                    self.player_jump_heights[player_id], 
                    height_above_ground
                )
            else:
                # Starting to land
                self.player_jump_states[player_id] = self.JUMP_STATES['LANDING']
                phase = 'landing'
                is_jumping = True
                
        elif current_state == self.JUMP_STATES['LANDING']:
            if height_above_ground < self.jump_threshold * 0.3:  # Close to ground
                # Landed
                self.player_jump_states[player_id] = self.JUMP_STATES['GROUND']
                self.player_jump_counts[player_id] += 1
                phase = 'ground'
            else:
                phase = 'landing'
                is_jumping = True
        
        return {
            'is_jumping': is_jumping,
            'phase': phase
        }
    
    def draw_jump_analysis(self, frame: np.ndarray, player_id: int, bbox: List[float], jump_info: Dict) -> np.ndarray:
        """Draw jump analysis visualization on frame"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color coding for jump phases
        colors = {
            'ground': (255, 255, 255),      # White
            'takeoff': (0, 255, 255),       # Cyan
            'airborne': (0, 0, 255),        # Red
            'landing': (255, 0, 255),       # Magenta
            'unknown': (128, 128, 128)      # Gray
        }
        
        jump_phase = jump_info.get('jump_phase', 'unknown')
        color = colors.get(jump_phase, (128, 128, 128))
        is_jumping = jump_info.get('is_jumping', False)
        
        # Draw jump indicator
        if is_jumping:
            # Draw jump indicator above player
            cv2.circle(frame, (int((x1 + x2) / 2), y1 - 15), 6, (0, 0, 255), -1)
            cv2.putText(frame, "JUMP!", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Jump stats below player
        stats_y = y2 + 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        
        # Display jump information
        jump_height = jump_info.get('jump_height', 0)
        confidence = jump_info.get('jump_confidence', 0)
        
        cv2.putText(frame, f"Jumps: {self.player_jump_counts.get(player_id, 0)}", 
                   (x1, stats_y), font, font_scale, color, 1)
        
        cv2.putText(frame, f"Height: {jump_height:.1f}px", 
                   (x1, stats_y + 12), font, font_scale, color, 1)
        
        cv2.putText(frame, f"Phase: {jump_phase}", 
                   (x1, stats_y + 24), font, font_scale, color, 1)
        
        cv2.putText(frame, f"Conf: {confidence:.2f}", 
                   (x1, stats_y + 36), font, font_scale, color, 1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, player_tracks: Dict, frame_num: int) -> Tuple[np.ndarray, Dict]:
        """Process a single frame for jump detection"""
        # Detect jumps using advanced tracking analysis
        jump_data = self.detect_jumps_from_tracking(player_tracks, frame_num, frame)
        
        # Draw analysis on frame
        annotated_frame = frame.copy()
        for player_id, jump_info in jump_data.items():
            if player_id in player_tracks:
                bbox = player_tracks[player_id]['bbox']
                annotated_frame = self.draw_jump_analysis(annotated_frame, player_id, bbox, jump_info)
        
        return annotated_frame, jump_data
    
    def get_player_jump_stats(self) -> Dict:
        """Get comprehensive jump statistics for all players"""
        stats = {}
        for player_id in self.player_jump_counts.keys():
            stats[player_id] = {
                'total_jumps': self.player_jump_counts[player_id],
                'max_jump_height': self.player_jump_heights[player_id],
                'current_state': self.player_jump_states[player_id]
            }
        return stats
    
    def save_jump_stats(self, output_dir: str = "output_data") -> str:
        """Save jump statistics to JSON file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_jump_stats_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        stats_data = {
            'timestamp': timestamp,
            'detection_method': 'Advanced OpenCV-based Jump Detection',
            'players': {}
        }
        
        for player_id in self.player_jump_counts.keys():
            stats_data['players'][f'player_{player_id}'] = {
                'player_id': int(player_id),
                'total_jumps': int(self.player_jump_counts[player_id]),
                'max_jump_height_pixels': float(self.player_jump_heights[player_id]),
                'final_state': int(self.player_jump_states[player_id])
            }
        
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        return filepath
    
    def reset(self):
        """Reset all tracking data"""
        self.player_positions_history.clear()
        self.player_velocities.clear()
        self.player_jump_states.clear()
        self.player_jump_counts.clear()
        self.player_jump_heights.clear()
        self.player_ground_levels.clear()
        self.player_bbox_heights.clear()