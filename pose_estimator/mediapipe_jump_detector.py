import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict, Optional
import json
import os
from datetime import datetime


class MediaPipeJumpDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, show_skeleton=True, show_ui=False):
        """
        Initialize MediaPipe Pose Detection for jump tracking
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            show_skeleton: Whether to display skeleton/bone structure (default: True)
            show_ui: Whether to display UI elements like text and indicators (default: False)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.show_skeleton = show_skeleton
        self.show_ui = show_ui
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Jump detection parameters
        self.jump_threshold = 0.03  # Minimum vertical displacement for jump (relative to image height)
        self.min_jump_duration = 3  # Minimum frames for a valid jump
        self.max_jump_duration = 20  # Maximum frames for a valid jump
        
        # Player tracking data
        self.player_pose_history = {}  # Store pose landmarks for each player
        self.player_jump_states = {}   # Current jump state for each player
        self.player_jump_counts = {}   # Total jumps for each player
        self.player_jump_heights = {}  # Maximum jump heights for each player
        self.player_ground_levels = {} # Ground level estimation for each player
        
        # Jump state definitions
        self.JUMP_STATES = {
            'GROUND': 0,
            'TAKEOFF': 1,
            'AIRBORNE': 2,
            'LANDING': 3
        }
    
    def detect_poses_in_player_crops(self, frame: np.ndarray, player_tracks: Dict) -> Dict:
        """
        Detect poses for all players in the current frame
        
        Args:
            frame: Current video frame
            player_tracks: Dictionary of player tracks with bboxes
            
        Returns:
            Dictionary with pose landmarks for each player
        """
        frame_poses = {}
        
        for player_id, track_info in player_tracks.items():
            bbox = track_info['bbox']
            
            # Extract player crop from frame
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding to ensure full body is captured
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            player_crop = frame[y1:y2, x1:x2]
            
            if player_crop.size == 0:
                continue
            
            # Convert BGR to RGB for MediaPipe
            rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
            
            # Process pose detection
            results = self.pose.process(rgb_crop)
            
            if results.pose_landmarks:
                # Convert landmarks back to original frame coordinates
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates in original frame
                    x = x1 + landmark.x * (x2 - x1)
                    y = y1 + landmark.y * (y2 - y1)
                    landmarks.append({
                        'x': x,
                        'y': y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                frame_poses[player_id] = {
                    'landmarks': landmarks,
                    'bbox': bbox,
                    'crop_offset': (x1, y1, x2, y2)  # Store crop coordinates for manual drawing
                }
        
        return frame_poses
    
    def update_jump_detection(self, player_id: int, landmarks: List[Dict], frame_num: int) -> Dict:
        """
        Update jump detection for a specific player
        
        Args:
            player_id: Player identifier
            landmarks: MediaPipe pose landmarks
            frame_num: Current frame number
            
        Returns:
            Dictionary with jump information
        """
        if player_id not in self.player_pose_history:
            self.player_pose_history[player_id] = []
            self.player_jump_states[player_id] = self.JUMP_STATES['GROUND']
            self.player_jump_counts[player_id] = 0
            self.player_jump_heights[player_id] = 0
            self.player_ground_levels[player_id] = []
        
        # Store pose history (keep last 30 frames)
        self.player_pose_history[player_id].append({
            'landmarks': landmarks,
            'frame': frame_num
        })
        
        if len(self.player_pose_history[player_id]) > 30:
            self.player_pose_history[player_id].pop(0)
        
        # Calculate key points for jump detection
        jump_info = self._analyze_jump_from_poses(player_id, landmarks, frame_num)
        
        return jump_info
    
    def _analyze_jump_from_poses(self, player_id: int, landmarks: List[Dict], frame_num: int) -> Dict:
        """
        Analyze jumping motion from pose landmarks
        
        Args:
            player_id: Player identifier
            landmarks: Current pose landmarks
            frame_num: Current frame number
            
        Returns:
            Dictionary with jump analysis results
        """
        # Key landmarks for jump detection
        # MediaPipe pose landmark indices:
        # 27, 28: Left and right ankle
        # 23, 24: Left and right hip
        # 11, 12: Left and right shoulder
        
        try:
            # Get foot positions (ankles)
            left_ankle = landmarks[27] if len(landmarks) > 27 else None
            right_ankle = landmarks[28] if len(landmarks) > 28 else None
            
            # Get hip positions
            left_hip = landmarks[23] if len(landmarks) > 23 else None
            right_hip = landmarks[24] if len(landmarks) > 24 else None
            
            if not all([left_ankle, right_ankle, left_hip, right_hip]):
                return {'is_jumping': False, 'jump_height': 0, 'jump_phase': 'unknown'}
            
            # Calculate average foot and hip positions
            avg_foot_y = (left_ankle['y'] + right_ankle['y']) / 2
            avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
            
            # Calculate leg length (hip to foot distance)
            leg_length = abs(avg_hip_y - avg_foot_y)
            
            # Update ground level estimation (when player is likely on ground)
            if len(self.player_pose_history[player_id]) >= 5:
                recent_foot_positions = [
                    (pose['landmarks'][27]['y'] + pose['landmarks'][28]['y']) / 2
                    for pose in self.player_pose_history[player_id][-5:]
                    if len(pose['landmarks']) > 28
                ]
                
                if recent_foot_positions:
                    # Ground level is the maximum (lowest on screen) foot position in recent frames
                    current_ground = max(recent_foot_positions)
                    self.player_ground_levels[player_id].append(current_ground)
                    
                    # Keep only recent ground level measurements
                    if len(self.player_ground_levels[player_id]) > 20:
                        self.player_ground_levels[player_id].pop(0)
            
            # Calculate ground level
            if self.player_ground_levels[player_id]:
                ground_level = np.percentile(self.player_ground_levels[player_id], 75)  # Use 75th percentile
            else:
                ground_level = avg_foot_y
            
            # Calculate vertical displacement from ground
            vertical_displacement = ground_level - avg_foot_y
            
            # Normalize by leg length to make it relative to player size
            if leg_length > 0:
                normalized_displacement = vertical_displacement / leg_length
            else:
                normalized_displacement = 0
            
            # Additional jump indicators
            knee_bend = self._calculate_knee_bend(landmarks)
            foot_separation = abs(left_ankle['x'] - right_ankle['x'])
            
            # Jump detection logic
            is_jumping = False
            jump_phase = 'ground'
            jump_height = max(0, vertical_displacement)
            
            current_state = self.player_jump_states[player_id]
            
            # State machine for jump detection
            if current_state == self.JUMP_STATES['GROUND']:
                # Look for takeoff: significant upward movement + knee bend
                if normalized_displacement > 0.15 and knee_bend > 0.3:
                    self.player_jump_states[player_id] = self.JUMP_STATES['TAKEOFF']
                    jump_phase = 'takeoff'
                    
            elif current_state == self.JUMP_STATES['TAKEOFF']:
                # Confirm airborne: continued upward movement
                if normalized_displacement > 0.2:
                    self.player_jump_states[player_id] = self.JUMP_STATES['AIRBORNE']
                    jump_phase = 'airborne'
                    is_jumping = True
                    # Update maximum jump height
                    self.player_jump_heights[player_id] = max(
                        self.player_jump_heights[player_id], 
                        vertical_displacement
                    )
                elif normalized_displacement < 0.1:
                    # False alarm, return to ground
                    self.player_jump_states[player_id] = self.JUMP_STATES['GROUND']
                    
            elif current_state == self.JUMP_STATES['AIRBORNE']:
                if normalized_displacement > 0.15:
                    jump_phase = 'airborne'
                    is_jumping = True
                    # Update maximum jump height
                    self.player_jump_heights[player_id] = max(
                        self.player_jump_heights[player_id], 
                        vertical_displacement
                    )
                else:
                    # Starting to descend
                    self.player_jump_states[player_id] = self.JUMP_STATES['LANDING']
                    jump_phase = 'landing'
                    is_jumping = True
                    
            elif current_state == self.JUMP_STATES['LANDING']:
                if normalized_displacement < 0.1:
                    # Landed, count the jump
                    self.player_jump_states[player_id] = self.JUMP_STATES['GROUND']
                    self.player_jump_counts[player_id] += 1
                    jump_phase = 'ground'
                else:
                    jump_phase = 'landing'
                    is_jumping = True
            
            return {
                'is_jumping': is_jumping,
                'jump_height': jump_height,
                'jump_phase': jump_phase,
                'vertical_displacement': vertical_displacement,
                'normalized_displacement': normalized_displacement,
                'knee_bend': knee_bend,
                'ground_level': ground_level,
                'foot_position': avg_foot_y,
                'leg_length': leg_length
            }
            
        except (IndexError, KeyError, TypeError) as e:
            return {'is_jumping': False, 'jump_height': 0, 'jump_phase': 'unknown', 'error': str(e)}
    
    def _calculate_knee_bend(self, landmarks: List[Dict]) -> float:
        """
        Calculate knee bend angle as an indicator of jumping motion
        
        Args:
            landmarks: Pose landmarks
            
        Returns:
            Knee bend factor (0-1, where 1 is maximum bend)
        """
        try:
            # Get leg keypoints
            left_hip = landmarks[23]
            left_knee = landmarks[25]
            left_ankle = landmarks[27]
            
            right_hip = landmarks[24]
            right_knee = landmarks[26]
            right_ankle = landmarks[28]
            
            # Calculate angles for both legs
            def calculate_angle(p1, p2, p3):
                """Calculate angle at p2 formed by p1-p2-p3"""
                v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
                v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                return np.degrees(angle)
            
            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Average angle
            avg_angle = (left_angle + right_angle) / 2
            
            # Convert to bend factor (180 degrees = straight leg = 0 bend, 90 degrees = max bend = 1)
            bend_factor = max(0, (180 - avg_angle) / 90)
            
            return min(1.0, bend_factor)
            
        except (IndexError, KeyError, TypeError, ZeroDivisionError):
            return 0.0
    
    def process_frame(self, frame: np.ndarray, player_tracks: Dict, frame_num: int) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for jump detection
        
        Args:
            frame: Input video frame
            player_tracks: Player tracking data
            frame_num: Current frame number
            
        Returns:
            Tuple of (annotated_frame, jump_data)
        """
        # Detect poses for all players
        frame_poses = self.detect_poses_in_player_crops(frame, player_tracks)
        
        # Analyze jumps for each player
        jump_data = {}
        annotated_frame = frame.copy()
        
        for player_id, pose_data in frame_poses.items():
            landmarks = pose_data['landmarks']
            
            # Update jump detection
            jump_info = self.update_jump_detection(player_id, landmarks, frame_num)
            jump_data[player_id] = jump_info
            
            # Draw pose and jump information
            annotated_frame = self.draw_jump_analysis(
                annotated_frame, 
                player_id, 
                pose_data, 
                jump_info
            )
        
        return annotated_frame, jump_data
    
    def draw_jump_analysis(self, frame: np.ndarray, player_id: int, pose_data: Dict, jump_info: Dict) -> np.ndarray:
        """
        Draw pose landmarks and jump information on the frame
        
        Args:
            frame: Input frame
            player_id: Player identifier
            pose_data: Pose detection results
            jump_info: Jump analysis results
            
        Returns:
            Annotated frame
        """
        landmarks = pose_data['landmarks']
        bbox = pose_data['bbox']
        
        # Draw full skeleton manually if enabled
        if self.show_skeleton and landmarks and len(landmarks) > 32:
            # Define MediaPipe pose connections (body skeleton structure)
            connections = [
                # Face
                (0, 1), (1, 2), (2, 3), (3, 7),
                (0, 4), (4, 5), (5, 6), (6, 8),
                # Upper body
                (9, 10),  # Mouth
                (11, 12),  # Shoulders
                (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Right arm
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Left arm
                (11, 23), (12, 24),  # Torso
                (23, 24),  # Hips
                # Lower body
                (23, 25), (25, 27), (27, 29), (27, 31),  # Right leg
                (24, 26), (26, 28), (28, 30), (28, 32),  # Left leg
            ]
            
            # Draw connections (bones)
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    
                    # Check visibility threshold
                    if start.get('visibility', 0) > 0.5 and end.get('visibility', 0) > 0.5:
                        cv2.line(frame, 
                                (int(start['x']), int(start['y'])), 
                                (int(end['x']), int(end['y'])), 
                                (255, 255, 255), 2)  # White lines
            
            # Draw keypoints
            for idx, landmark in enumerate(landmarks):
                if landmark.get('visibility', 0) > 0.5:
                    # Color code different body parts
                    if idx <= 10:  # Face
                        color = (0, 255, 255)  # Cyan
                    elif idx <= 16:  # Arms
                        color = (0, 255, 0)  # Green
                    elif idx <= 24:  # Torso
                        color = (255, 0, 0)  # Blue
                    else:  # Legs
                        color = (0, 165, 255)  # Orange
                    
                    cv2.circle(frame, 
                              (int(landmark['x']), int(landmark['y'])), 
                              4, color, -1)
        
        # Only draw UI elements if enabled
        if self.show_ui:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Jump status text
            jump_phase = jump_info.get('jump_phase', 'unknown')
            jump_height = jump_info.get('jump_height', 0)
            is_jumping = jump_info.get('is_jumping', False)
            
            # Color coding for jump phases
            colors = {
                'ground': (255, 255, 255),      # White
                'takeoff': (0, 255, 255),       # Cyan
                'airborne': (0, 0, 255),        # Red
                'landing': (255, 0, 255),       # Magenta
                'unknown': (128, 128, 128)      # Gray
            }
            
            color = colors.get(jump_phase, (128, 128, 128))
            
            # Jump indicator
            if is_jumping:
                # Draw jump indicator above player
                cv2.circle(frame, (int((x1 + x2) / 2), y1 - 20), 8, (0, 0, 255), -1)
                cv2.putText(frame, "JUMP!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Jump stats text below player
            stats_y = y2 + 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            
            # Jump count and height
            cv2.putText(frame, f"Jumps: {self.player_jump_counts.get(player_id, 0)}", 
                       (x1, stats_y), font, font_scale, color, 1)
            
            cv2.putText(frame, f"Height: {jump_height:.1f}px", 
                       (x1, stats_y + 15), font, font_scale, color, 1)
            
            cv2.putText(frame, f"Phase: {jump_phase}", 
                       (x1, stats_y + 30), font, font_scale, color, 1)
        
        return frame
    
    def get_player_jump_stats(self) -> Dict:
        """
        Get comprehensive jump statistics for all players
        
        Returns:
            Dictionary with jump stats for each player
        """
        stats = {}
        for player_id in self.player_jump_counts.keys():
            stats[player_id] = {
                'total_jumps': self.player_jump_counts[player_id],
                'max_jump_height': self.player_jump_heights[player_id],
                'current_state': self.player_jump_states[player_id]
            }
        return stats
    
    def save_jump_stats(self, output_dir: str = "output_data") -> str:
        """
        Save jump statistics to JSON file
        
        Args:
            output_dir: Output directory for the stats file
            
        Returns:
            Path to the saved file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mediapipe_jump_stats_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        stats_data = {
            'timestamp': timestamp,
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
        self.player_pose_history.clear()
        self.player_jump_states.clear()
        self.player_jump_counts.clear()
        self.player_jump_heights.clear()
        self.player_ground_levels.clear()