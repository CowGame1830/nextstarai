from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
# import torchreid
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from player_stats import PlayerStatsTracker
from pose_estimator.advanced_jump_detector_clean import AdvancedJumpDetector
from minimap_generator import MinimapGenerator
from torchreid.utils import FeatureExtractor
import torch
from fake_tracker import FakeTracker

def main():
    # Read Video
    video_frames = read_video('input_videos/10secVideo.mp4')
    video_frames = video_frames[:100]   # <-- เอาแค่ 10 เฟรมแรก
    print(f"จำนวนเฟรมใน video_frames = {len(video_frames)}")

    # Initialize Tracker    
    tracker = Tracker('models/model_2_0.pt')
        
    # tracker = FakeTracker(image_dir="image", label_dir="label")

    tracks = tracker.get_object_tracks(video_frames,
                                           read_from_stub=False,
                                           stub_path='stubs/track_stubs.pkl')
    print(f"[DEBUG] จำนวนเฟรมใน tracks['players'] = {len(tracks['players'])}")
    print(f"[DEBUG] จำนวนเฟรมใน tracks['ball'] = {len(tracks['ball'])}")
    print(f"[DEBUG] จำนวนเฟรมใน tracks['referees'] = {len(tracks['referees'])}")
    # total_frames = 200  # <-- ปรับตามจำนวนไฟล์ใน folder image ของคุณ
    # tracker = FakeTracker(image_dir="image", label_dir="label", total_frames=total_frames)
    # tracks = tracker.get_object_tracks()
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Debug: save positions for all players in all frames to txt file
    if len(tracks['players']) > 0:
        import os
        from datetime import datetime
        
        os.makedirs("output_data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = f"output_data/player_positions_debug_{timestamp}.txt"
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PLAYER POSITIONS DEBUG LOG (ALL FRAMES)\n")
            f.write("=" * 80 + "\n\n")
            
            for frame_num, player_track in enumerate(tracks['players']):
                f.write(f"\n{'='*80}\n")
                f.write(f"FRAME {frame_num} - Total Players: {len(player_track)}\n")
                f.write(f"{'='*80}\n")
                
                for pid in sorted(player_track.keys()):
                    info = player_track[pid]
                    bbox = info.get('bbox')
                    pos_raw = info.get('position')
                    pos_trans = info.get('position_transformed')
                    
                    f.write(f"\n  Player ID: {pid}\n")
                    f.write(f"    bbox: {bbox}\n")
                    f.write(f"    position: {pos_raw}\n")
                    f.write(f"    position_transformed: {pos_trans}\n")
                    
                    # Add status indicator
                    if pos_trans is None:
                        f.write(f"    STATUS: ⚠️ OUTSIDE POLYGON (No transformation)\n")
                    else:
                        f.write(f"    STATUS: ✓ Valid transformation\n")
        
        print(f"\n[DEBUG] Player positions saved to: {debug_file}")

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator with advanced jump detection
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.set_video_frames(video_frames)  # Set frames for jump detection
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # Initialize advanced jump detector for enhanced jump tracking
    jump_detector = AdvancedJumpDetector()
    print("Advanced jump detection initialized")

    # Initialize Player Statistics Tracker
    player_stats_tracker = PlayerStatsTracker(frame_rate=24)
    player_stats_tracker.update_player_stats(tracks)
    
    # Initialize Minimap Generator
    minimap_generator = MinimapGenerator(minimap_width=350, minimap_height=230)
    print("Minimap generator initialized")

    # Assign Player Teams
    team_assigner = TeamAssigner()
    
        # -------------------------
    # 11. OSNet feature extractor
    # -------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    osnet_extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='osnet_x1_0_imagenet.pth',
        device=device
    )
    print("OSNet feature extractor initialized")

    # Extract OSNet feature for all players
    for frame_num, player_track in enumerate(tracks['players']):
        frame = video_frames[frame_num]
        for player_id, pdata in player_track.items():
            x1, y1, x2, y2 = map(int, pdata['bbox'])
            player_crop = frame[y1:y2, x1:x2]
            try:
                feature = osnet_extractor([player_crop])[0]  # feature vector
                tracks['players'][frame_num][player_id]['osnet_feature'] = feature
            except Exception as e:
                print(f"[Warning] Cannot extract OSNet feature for Player {player_id} in frame {frame_num}: {e}")
                tracks['players'][frame_num][player_id]['osnet_feature'] = None


    # Find first frame with players for team color assignment
    first_frame_with_players = None
    for frame_idx, player_track in enumerate(tracks['players']):
        if len(player_track) > 0:
            first_frame_with_players = frame_idx
            break
    
    if first_frame_with_players is not None:
        try:
            team_assigner.assign_team_color(video_frames[first_frame_with_players], 
                                          tracks['players'][first_frame_with_players])
            print(f"Team colors assigned using frame {first_frame_with_players}")
        except Exception as e:
            print(f"Warning: Could not assign team colors: {e}")
            # Set default team colors
            team_assigner.team_colors = {1: (255, 0, 0), 2: (0, 0, 255)}
    else:
        print("Warning: No players found in any frame for team assignment")
        team_assigner.team_colors = {1: (255, 0, 0), 2: (0, 0, 255)}
    
    # Set minimap team colors to match assigned team colors
    if hasattr(team_assigner, 'team_colors') and team_assigner.team_colors:
        # Convert BGR to RGB for minimap (OpenCV uses BGR, minimap uses RGB)
        team1_color = team_assigner.team_colors.get(1, (255, 0, 0))  # Default red
        team2_color = team_assigner.team_colors.get(2, (0, 0, 255))  # Default blue
        # Convert BGR to RGB
        team1_rgb = (team1_color[2], team1_color[1], team1_color[0])
        team2_rgb = (team2_color[2], team2_color[1], team2_color[0])
        minimap_generator.set_team_colors(team1_rgb, team2_rgb)
        print(f"Minimap team colors set: Team 1: {team1_rgb}, Team 2: {team2_rgb}")
    
    # Assign teams to all players
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            try:
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                     track['bbox'],
                                                     player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            except Exception as e:
                # Assign default team if assignment fails
                tracks['players'][frame_num][player_id]['team'] = 1
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(1, (255, 0, 0))

    
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # Use previous team control or default to team 1 if no previous data
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)  # Default to team 1
    
    team_ball_control = np.array(team_ball_control)

    # Debug: list player IDs และ bbox ทุก frame
    # print("\n[DEBUG] ตรวจสอบ player IDs และ bbox ทุก frame")
    # for frame_num, player_track in enumerate(tracks['players']):
    #     print(f"\nFrame {frame_num} - จำนวนผู้เล่น: {len(player_track)}")
    #     for pid, pdata in player_track.items():
    #         bbox = pdata['bbox']
    #         team = pdata.get('team', None)
    #         has_ball = pdata.get('has_ball', False)
    #         print(f"  PlayerID {pid}: bbox={bbox}, team={team}, has_ball={has_ball}")

    all_player_ids = set()

    for frame_num, player_track in enumerate(tracks['players']):
        all_player_ids.update(player_track.keys())

    print("\n[DEBUG] Player IDs ทั้งหมดที่ detect ได้ตลอดคลิป:")
    print(all_player_ids)
    print(f"จำนวน Player IDs ทั้งหมด = {len(all_player_ids)}")

    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # Process advanced jump detection for all frames FIRST
    print("Processing advanced jump detection...")
    
    for frame_num in range(len(tracks['players'])):
        if frame_num < len(video_frames):
            # Get player tracks for current frame
            current_player_tracks = {}
            for player_id, track_info in tracks['players'][frame_num].items():
                current_player_tracks[player_id] = {'bbox': track_info['bbox']}
            
            # Process jump detection
            _, jump_data = jump_detector.process_frame(video_frames[frame_num], current_player_tracks, frame_num)
            
            # Update tracks with advanced jump data
            for player_id, jump_info in jump_data.items():
                if player_id in tracks['players'][frame_num]:
                    tracks['players'][frame_num][player_id]['advanced_jump'] = jump_info.get('is_jumping', False)
                    tracks['players'][frame_num][player_id]['jump_phase'] = jump_info.get('jump_phase', 'unknown')
                    tracks['players'][frame_num][player_id]['jump_height_advanced'] = jump_info.get('jump_height', 0)
                    tracks['players'][frame_num][player_id]['jump_confidence'] = jump_info.get('jump_confidence', 0)

    ## Draw Speed and Distance (now with updated jump data)
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    ## Add Minimap to existing frames (integrated into main output)
    print("Adding minimap overlay to video frames...")
    output_video_frames = minimap_generator.draw_minimap_with_stats(
        output_video_frames, 
        tracks, 
        position='bottom_center',
        team_ball_control=team_ball_control
    )
    print("Minimap overlay completed - integrated into main video")
    
    # Save advanced jump detection statistics
    jump_stats_file = jump_detector.save_jump_stats("output_data")
    print(f"Advanced jump statistics saved to: {jump_stats_file}")
    
    # Save player statistics data to JSON
    stats_file = player_stats_tracker.save_stats_to_file("output_data")
    print(f"Player statistics saved to: {stats_file}")
    
    # Save enhanced speed and distance statistics with advanced jump data
    enhanced_stats_file = speed_and_distance_estimator.save_enhanced_stats_to_json("output_data")
    print(f"Enhanced statistics with advanced jump detection saved to: {enhanced_stats_file}")
    
    # Save speed calculation debug log to txt file
    speed_debug_file = speed_and_distance_estimator.save_speed_debug_log("output_data")
    if speed_debug_file:
        print(f"Speed calculation debug log saved to: {speed_debug_file}")

    # Save video with all features including minimap (same filename as before)
    save_video(output_video_frames, 'output_videos/output_video.avi')

    # Print advanced jump detection summary
    jump_stats = jump_detector.get_player_jump_stats()
    print("\n=== Advanced Jump Detection Summary ===")
    for player_id, stats in jump_stats.items():
        print(f"Player {player_id}: {stats['total_jumps']} jumps detected, Max height: {stats['max_jump_height']:.1f}px")
    print("==========================================\n")


###############################################################################


#test 1 frame
def main3():
    # # Read Video
    # # video_frames = read_video('input_videos/CHEvLIV1.mp4')
    # video_frames = read_video('input_videos/MCIvsEVE1080p2mp4.mp4')

    # # Initialize Tracker
    # tracker = Tracker('models/best.pt')

    # tracks = tracker.get_object_tracks(video_frames,
    #                                    read_from_stub=True,
    #                                    stub_path='stubs/track_stubs.pkl')

    # # แสดง bbox เฟรมแรก
    # first_frame_players = tracks['players'][0]  # เฟรมแรก
    # print("=== BBOX ของผู้เล่นในเฟรมแรก ===")
    # for player_id, player_info in first_frame_players.items():
    #     print(f"Player {player_id}: {player_info['bbox']}")

    # # วาด bbox ลงบนภาพ
    # frame = video_frames[0].copy()
    # for player_id, player_info in first_frame_players.items():
    #     bbox = player_info['bbox']
    #     x1, y1, x2, y2 = map(int, bbox)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    #     cv2.putText(frame, str(player_id), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # # เซฟภาพ
    # output_path = "output_test/frame1080_bbox.jpg"
    # cv2.imwrite(output_path, frame)
    # print(f"Saved frame with bbox to {output_path}")
    # print("Frame shape:", frame.shape)
    def print_tracks_structure(tracks):
        print("\n=== KEYS ใน tracks ===")
        print(tracks.keys())

        print("\n=== ตัวอย่างจำนวนเฟรมที่มีในแต่ละ category ===")
        for key in tracks.keys():
            print(f"{key}: {len(tracks[key])} frames")

        print("\n=== ตัวอย่างข้อมูลเฟรมแรกของแต่ละ category ===")
        for key in tracks.keys():
            print(f"\n--- {key} frame[0] ---")
            print(tracks[key][0])
    video_frames = read_video('input_videos/MCIvsEVE1080p2mp4.mp4')
    tracker = Tracker('models/model_2_0.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    print_tracks_structure(tracks)



def debug():
    IMAGE_PATH = "image/frame_0000.jpg"
    LABEL_PATH = "label/frame_0000.txt"

    def show_first_frame(image_path, label_path):
        # โหลดภาพ
        frame = cv2.imread(image_path)
        H, W = frame.shape[:2]

        # โหลด label YOLO normalized
        bboxes = []
        with open(label_path, "r") as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.strip().split())

                # แปลง YOLO -> pixel
                x1 = int((cx - w/2) * W)
                y1 = int((cy - h/2) * H)
                x2 = int((cx + w/2) * W)
                y2 = int((cy + h/2) * H)

                bboxes.append((int(cls), x1, y1, x2, y2))

        # ----- ปริ้น TXT -----
        print("===== TXT ของเฟรมแรก (pixel) =====")
        for cls, x1, y1, x2, y2 in bboxes:
            print(f"Class {cls}: [{x1}, {y1}, {x2}, {y2}]")

        # ----- วาด BBOX -----
        for cls, x1, y1, x2, y2 in bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(cls), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ----- แสดงภาพ -----
        print("\n===== IMAGE: เฟรมแรกพร้อม BBOX =====")
        cv2.imshow("Frame First", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    show_first_frame(IMAGE_PATH, LABEL_PATH)


if __name__ == '__main__':
    main()
