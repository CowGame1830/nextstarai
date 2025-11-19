import cv2
import numpy as np
import math


class MinimapGenerator:
    def __init__(self, minimap_width=300, minimap_height=200):
        """
        Initialize minimap generator
        
        Args:
            minimap_width (int): Width of the minimap
            minimap_height (int): Height of the minimap
        """
        self.minimap_width = minimap_width
        self.minimap_height = minimap_height
        
        # Football field dimensions (standard FIFA dimensions in meters)
        self.field_length = 105  # Length of the field in meters
        self.field_width = 68    # Width of the field in meters
        
        # Minimap field dimensions with margins
        self.field_margin = 20
        self.field_draw_width = self.minimap_width - (2 * self.field_margin)
        self.field_draw_height = self.minimap_height - (2 * self.field_margin)
        
        # Colors for different elements
        self.colors = {
            'field': (34, 139, 34),      # Forest Green
            'lines': (255, 255, 255),    # White
            'team1': (255, 100, 100),    # Light Red
            'team2': (100, 100, 255),    # Light Blue
            'ball': (255, 255, 0),       # Yellow
            'referee': (128, 128, 128),  # Gray
            'background': (20, 60, 20)   # Dark Green
        }
        
        # Player size on minimap
        self.player_radius = 4
        self.ball_radius = 3
        
        # Ball trail for dynamic effect
        self.ball_trail = []
        self.max_trail_length = 8
        
        # Player with ball highlighting
        self.player_with_ball_pulse = 0
        
        # Enhanced FIFA features
        self.show_player_numbers = True
        self.show_speed_indicators = True
        self.show_possession_stats = True
        self.frame_count = 0
        
        # Create base field template
        self.field_template = self._create_field_template()
    
    def _create_field_template(self):
        """Create a template of the football field"""
        field = np.full((self.minimap_height, self.minimap_width, 3), 
                       self.colors['background'], dtype=np.uint8)
        
        # Draw field background
        field_rect = (
            self.field_margin,
            self.field_margin,
            self.field_draw_width,
            self.field_draw_height
        )
        cv2.rectangle(field, 
                     (field_rect[0], field_rect[1]),
                     (field_rect[0] + field_rect[2], field_rect[1] + field_rect[3]),
                     self.colors['field'], -1)
        
        # Draw field outline
        cv2.rectangle(field,
                     (field_rect[0], field_rect[1]),
                     (field_rect[0] + field_rect[2], field_rect[1] + field_rect[3]),
                     self.colors['lines'], 2)
        
        # Draw center line
        center_x = field_rect[0] + field_rect[2] // 2
        cv2.line(field,
                (center_x, field_rect[1]),
                (center_x, field_rect[1] + field_rect[3]),
                self.colors['lines'], 2)
        
        # Draw center circle
        center_y = field_rect[1] + field_rect[3] // 2
        circle_radius = int(field_rect[3] * 0.15)  # Proportional to field height
        cv2.circle(field, (center_x, center_y), circle_radius, self.colors['lines'], 2)
        
        # Draw penalty areas
        penalty_width = int(field_rect[2] * 0.25)  # 25% of field width
        penalty_height = int(field_rect[3] * 0.4)   # 40% of field height
        
        # Left penalty area
        left_penalty_y = field_rect[1] + (field_rect[3] - penalty_height) // 2
        cv2.rectangle(field,
                     (field_rect[0], left_penalty_y),
                     (field_rect[0] + penalty_width, left_penalty_y + penalty_height),
                     self.colors['lines'], 2)
        
        # Right penalty area
        right_penalty_x = field_rect[0] + field_rect[2] - penalty_width
        cv2.rectangle(field,
                     (right_penalty_x, left_penalty_y),
                     (field_rect[0] + field_rect[2], left_penalty_y + penalty_height),
                     self.colors['lines'], 2)
        
        # Draw goal areas (smaller rectangles)
        goal_width = int(field_rect[2] * 0.1)   # 10% of field width
        goal_height = int(field_rect[3] * 0.2)  # 20% of field height
        
        # Left goal area
        left_goal_y = field_rect[1] + (field_rect[3] - goal_height) // 2
        cv2.rectangle(field,
                     (field_rect[0], left_goal_y),
                     (field_rect[0] + goal_width, left_goal_y + goal_height),
                     self.colors['lines'], 2)
        
        # Right goal area
        right_goal_x = field_rect[0] + field_rect[2] - goal_width
        cv2.rectangle(field,
                     (right_goal_x, left_goal_y),
                     (field_rect[0] + field_rect[2], left_goal_y + goal_height),
                     self.colors['lines'], 2)
        
        return field
    
    def _transform_coordinates(self, transformed_position):
        """
        Transform world coordinates to minimap coordinates
        
        Args:
            transformed_position (tuple): (x, y) position in transformed coordinates
        
        Returns:
            tuple: (x, y) position on minimap
        """
        if transformed_position is None:
            return None
            
        # Normalize coordinates (assuming transformed coordinates are in meters)
        # The view transformer should provide coordinates in meters relative to field
        x_ratio = transformed_position[0] / self.field_length
        y_ratio = transformed_position[1] / self.field_width
        
        # Clamp to field boundaries
        x_ratio = max(0, min(1, x_ratio))
        y_ratio = max(0, min(1, y_ratio))
        
        # Convert to minimap pixel coordinates
        minimap_x = int(self.field_margin + x_ratio * self.field_draw_width)
        minimap_y = int(self.field_margin + y_ratio * self.field_draw_height)
        
        return (minimap_x, minimap_y)
    
    def generate_minimap(self, tracks, frame_num, team_ball_control=None):
        """
        Generate minimap for a specific frame
        
        Args:
            tracks (dict): Tracking data for all objects
            frame_num (int): Current frame number
            team_ball_control (list): Ball possession data per frame
        
        Returns:
            numpy.ndarray: Minimap image
        """
        # Start with field template
        minimap = self.field_template.copy()
        
        # Add enhanced ball possession statistics if available
        if team_ball_control is not None and frame_num < len(team_ball_control):
            possession_team = team_ball_control[frame_num]
            possession_color = self.colors['team1'] if possession_team == 1 else self.colors['team2']
            
            # Draw possession indicator with percentage
            if self.show_possession_stats and frame_num > 0:
                # Calculate possession percentage up to current frame
                team1_frames = np.sum(team_ball_control[:frame_num+1] == 1)
                team2_frames = np.sum(team_ball_control[:frame_num+1] == 2)
                total_frames = team1_frames + team2_frames
                
                if total_frames > 0:
                    team1_pct = (team1_frames / total_frames) * 100
                    team2_pct = (team2_frames / total_frames) * 100
                    
                    # Draw possession stats in top corners
                    cv2.putText(minimap, f"Team 1: {team1_pct:.0f}%", 
                               (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['team1'], 1)
                    cv2.putText(minimap, f"Team 2: {team2_pct:.0f}%", 
                               (self.minimap_width - 80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors['team2'], 1)
            
            # Draw current possession indicator
            cv2.putText(minimap, f"Ball: Team {possession_team}", 
                       (5, self.minimap_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, possession_color, 1)
        
        # Draw players
        if frame_num < len(tracks['players']):
            player_tracks = tracks['players'][frame_num]
            
            for player_id, track_info in player_tracks.items():
                # Get transformed position
                if 'position_transformed' in track_info:
                    transformed_pos = track_info['position_transformed']
                    minimap_pos = self._transform_coordinates(transformed_pos)
                    
                    if minimap_pos:
                        # Determine team color
                        team = track_info.get('team', 1)
                        if team == 1:
                            color = self.colors['team1']
                        elif team == 2:
                            color = self.colors['team2']
                        else:
                            color = self.colors['team1']  # Default to team 1
                        
                        # Check if player has the ball for special highlighting
                        has_ball = track_info.get('has_ball', False)
                        speed = track_info.get('speed', 0) if 'speed' in track_info else 0
                        
                        # Speed-based player size (FIFA-style)
                        if self.show_speed_indicators and speed > 0:
                            speed_factor = min(speed / 25.0, 1.5)  # Scale based on speed (max 25 km/h)
                            dynamic_radius = int(self.player_radius * (0.8 + speed_factor * 0.4))
                        else:
                            dynamic_radius = self.player_radius
                        
                        if has_ball:
                            # Pulse effect for player with ball
                            self.player_with_ball_pulse = (self.player_with_ball_pulse + 1) % 20
                            pulse_size = 2 + int(2 * math.sin(self.player_with_ball_pulse * 0.3))
                            
                            # Draw pulsing ring around player with ball
                            cv2.circle(minimap, minimap_pos, dynamic_radius + pulse_size, (255, 255, 0), 2)
                        
                        # Speed indicator (color intensity based on speed)
                        if self.show_speed_indicators and speed > 15:  # High speed threshold
                            # Draw speed trail behind player
                            trail_length = min(int(speed / 5), 8)
                            for i in range(trail_length):
                                alpha = 1.0 - (i / trail_length)
                                trail_color = tuple(int(c * alpha) for c in color)
                                trail_pos = (minimap_pos[0] - i, minimap_pos[1])
                                cv2.circle(minimap, trail_pos, 1, trail_color, -1)
                        
                        # Draw player dot with dynamic size
                        cv2.circle(minimap, minimap_pos, dynamic_radius, color, -1)
                        
                        # Draw player outline for better visibility
                        cv2.circle(minimap, minimap_pos, dynamic_radius, (255, 255, 255), 1)
                        
                        # Draw player number (FIFA-style)
                        if self.show_player_numbers and self.minimap_width > 250:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.25
                            
                            # Convert player_id to jersey number (1-11 for each team)
                            jersey_number = ((player_id - 1) % 11) + 1
                            number_text = str(jersey_number)
                            
                            # Position number in center of player dot
                            text_size = cv2.getTextSize(number_text, font, font_scale, 1)[0]
                            text_pos = (minimap_pos[0] - text_size[0]//2, minimap_pos[1] + text_size[1]//2)
                            
                            # Draw number with contrasting color
                            number_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                            cv2.putText(minimap, number_text, text_pos, font, font_scale, number_color, 1)
        
        # Draw ball with trail effect
        if frame_num < len(tracks['ball']) and len(tracks['ball'][frame_num]) > 0:
            # Get first ball (usually there's only one)
            ball_track = list(tracks['ball'][frame_num].values())[0]
            
            if 'position_transformed' in ball_track:
                transformed_pos = ball_track['position_transformed']
                minimap_pos = self._transform_coordinates(transformed_pos)
                
                if minimap_pos:
                    # Add current position to trail
                    self.ball_trail.append(minimap_pos)
                    if len(self.ball_trail) > self.max_trail_length:
                        self.ball_trail.pop(0)
                    
                    # Draw ball trail (fading effect)
                    for i, trail_pos in enumerate(self.ball_trail[:-1]):  # Don't include current position
                        alpha = (i + 1) / len(self.ball_trail)  # Fading alpha
                        trail_radius = int(self.ball_radius * alpha)
                        trail_color = tuple(int(c * alpha) for c in self.colors['ball'])
                        if trail_radius > 0:
                            cv2.circle(minimap, trail_pos, trail_radius, trail_color, -1)
                    
                    # Draw current ball position (brightest)
                    ball_color = self.colors['ball']
                    cv2.circle(minimap, minimap_pos, self.ball_radius, ball_color, -1)
                    cv2.circle(minimap, minimap_pos, self.ball_radius + 1, (255, 255, 255), 1)
        
        # Draw referees
        if 'referees' in tracks and frame_num < len(tracks['referees']):
            referee_tracks = tracks['referees'][frame_num]
            
            for referee_id, track_info in referee_tracks.items():
                if 'position_transformed' in track_info:
                    transformed_pos = track_info['position_transformed']
                    minimap_pos = self._transform_coordinates(transformed_pos)
                    
                    if minimap_pos:
                        # Draw referee as diamond shape
                        referee_color = self.colors['referee']
                        points = np.array([
                            [minimap_pos[0], minimap_pos[1] - 4],  # Top
                            [minimap_pos[0] + 4, minimap_pos[1]],  # Right
                            [minimap_pos[0], minimap_pos[1] + 4],  # Bottom
                            [minimap_pos[0] - 4, minimap_pos[1]]   # Left
                        ], np.int32)
                        cv2.fillPoly(minimap, [points], referee_color)
                        cv2.polylines(minimap, [points], True, (255, 255, 255), 1)
        
        return minimap
    
    def add_minimap_to_frame(self, frame, minimap, position='bottom_center'):
        """
        Add minimap overlay to the main frame
        
        Args:
            frame (numpy.ndarray): Main video frame
            minimap (numpy.ndarray): Generated minimap
            position (str): Position of minimap on frame
        
        Returns:
            numpy.ndarray: Frame with minimap overlay
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate minimap position
        if position == 'bottom_center':
            start_x = (frame_width - self.minimap_width) // 2
            start_y = frame_height - self.minimap_height - 20  # 20px margin from bottom
        elif position == 'bottom_right':
            start_x = frame_width - self.minimap_width - 20
            start_y = frame_height - self.minimap_height - 20
        elif position == 'bottom_left':
            start_x = 20
            start_y = frame_height - self.minimap_height - 20
        elif position == 'top_right':
            start_x = frame_width - self.minimap_width - 20
            start_y = 20
        elif position == 'top_left':
            start_x = 20
            start_y = 20
        else:  # Default to bottom_center
            start_x = (frame_width - self.minimap_width) // 2
            start_y = frame_height - self.minimap_height - 20
        
        # Ensure minimap fits within frame bounds
        start_x = max(0, min(start_x, frame_width - self.minimap_width))
        start_y = max(0, min(start_y, frame_height - self.minimap_height))
        
        # Create a copy of the frame to avoid modifying the original
        output_frame = frame.copy()
        
        # Add semi-transparent border around minimap for better visibility
        border_thickness = 3
        border_color = (255, 255, 255)
        
        # Draw border
        cv2.rectangle(output_frame,
                     (start_x - border_thickness, start_y - border_thickness),
                     (start_x + self.minimap_width + border_thickness, 
                      start_y + self.minimap_height + border_thickness),
                     border_color, border_thickness)
        
        # Add minimap to frame
        end_x = start_x + self.minimap_width
        end_y = start_y + self.minimap_height
        
        output_frame[start_y:end_y, start_x:end_x] = minimap
        
        return output_frame
    
    def draw_minimap_with_stats(self, frames, tracks, position='bottom_center', team_ball_control=None):
        """
        Process all frames and add minimap overlay
        
        Args:
            frames (list): List of video frames
            tracks (dict): Tracking data for all objects
            position (str): Position of minimap on frame
            team_ball_control (list): Ball possession data per frame
        
        Returns:
            list: Frames with minimap overlay
        """
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            # Update frame count for animations
            self.frame_count = frame_num
            
            # Generate minimap for current frame
            minimap = self.generate_minimap(tracks, frame_num, team_ball_control)
            
            # Add minimap to frame
            frame_with_minimap = self.add_minimap_to_frame(frame, minimap, position)
            
            output_frames.append(frame_with_minimap)
        
        return output_frames
    
    def set_minimap_size(self, width, height):
        """
        Update minimap dimensions and regenerate field template
        
        Args:
            width (int): New minimap width
            height (int): New minimap height
        """
        self.minimap_width = width
        self.minimap_height = height
        
        # Recalculate field dimensions
        self.field_draw_width = self.minimap_width - (2 * self.field_margin)
        self.field_draw_height = self.minimap_height - (2 * self.field_margin)
        
        # Regenerate field template
        self.field_template = self._create_field_template()
    
    def set_team_colors(self, team1_color, team2_color):
        """
        Set custom team colors
        
        Args:
            team1_color (tuple): RGB color for team 1
            team2_color (tuple): RGB color for team 2
        """
        self.colors['team1'] = team1_color
        self.colors['team2'] = team2_color