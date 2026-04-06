from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# NMS imports
try:
    from torchvision.ops import nms
except ImportError:
    nms = None

class Tracker:
    def __init__(self, model_path, frame_rate=24, detection_conf=0.35, imgsz=1280, batch_size=20, nms_threshold=0.4):
        self.model = YOLO(model_path)
        self.frame_rate = frame_rate
        self.detection_conf = float(detection_conf)  # Increased from 0.1 to 0.35 for better confidence
        self.imgsz = int(imgsz)
        self.batch_size = int(batch_size)
        self.nms_threshold = float(nms_threshold)  # IoU threshold for NMS, reduced to 0.4 for stricter NMS

        # Configure ByteTrack for stronger ID persistence under short occlusions.
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.2,
                minimum_matching_threshold=0.75,
                lost_track_buffer=60,
                frame_rate=frame_rate,
            )
        except TypeError:
            # Fallback for older/newer supervision signatures.
            self.tracker = sv.ByteTrack()

    def _custom_nms(self, boxes, scores, iou_threshold=0.5):
        """Custom NMS implementation using IoU calculation.
        
        Args:
            boxes: Array of bounding boxes [N, 4] in format [x1, y1, x2, y2]
            scores: Array of confidence scores [N]
            iou_threshold: IoU threshold for suppression
            
        Returns:
            keep_indices: Indices of boxes to keep after NMS
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        # Sort by confidence score in descending order
        sorted_indices = np.argsort(-scores)
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Keep the box with highest confidence
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            iou_scores = np.array([
                self._bbox_iou(current_box, box) 
                for box in remaining_boxes
            ])
            
            # Keep only boxes with IoU below threshold
            keep_mask = iou_scores < iou_threshold
            sorted_indices = sorted_indices[1:][keep_mask]
        
        return np.array(keep_indices, dtype=np.int32)

    def _apply_nms_to_detections(self, detection_supervision, iou_threshold=None):
        """Apply NMS to supervision detections to remove duplicates.
        
        Args:
            detection_supervision: Supervision Detections object
            iou_threshold: IoU threshold (uses self.nms_threshold if None)
            
        Returns:
            Filtered detections
        """
        if iou_threshold is None:
            iou_threshold = self.nms_threshold
        
        if detection_supervision.xyxy is None or len(detection_supervision.xyxy) == 0:
            return detection_supervision
        
        boxes = detection_supervision.xyxy
        scores = detection_supervision.confidence if detection_supervision.confidence is not None else np.ones(len(boxes))
        
        # Try using torchvision NMS if available
        if nms is not None:
            try:
                import torch
                boxes_tensor = torch.from_numpy(boxes).float()
                scores_tensor = torch.from_numpy(scores).float()
                keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold).numpy()
            except Exception as e:
                print(f"Warning: Torchvision NMS failed ({e}), using custom NMS")
                keep_indices = self._custom_nms(boxes, scores, iou_threshold)
        else:
            # Fallback to custom NMS implementation
            keep_indices = self._custom_nms(boxes, scores, iou_threshold)
        
        # Filter detections
        if len(keep_indices) < len(boxes):
            filtered_detections = detection_supervision[keep_indices]
            return filtered_detections
        
        return detection_supervision

    def _bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0
        return inter_area / union

    def _bbox_motion_score(self, prev_box, curr_box):
        prev_cx, prev_cy = get_center_of_bbox(prev_box)
        curr_cx, curr_cy = get_center_of_bbox(curr_box)

        prev_w = max(1.0, prev_box[2] - prev_box[0])
        prev_h = max(1.0, prev_box[3] - prev_box[1])
        curr_w = max(1.0, curr_box[2] - curr_box[0])
        curr_h = max(1.0, curr_box[3] - curr_box[1])

        scale = max(prev_w, prev_h, curr_w, curr_h)
        center_dist = np.hypot(curr_cx - prev_cx, curr_cy - prev_cy)

        # Convert distance to a [0,1] similarity score.
        return max(0.0, 1.0 - (center_dist / (scale * 2.5)))

    def _bbox_size_similarity(self, box_a, box_b):
        aw = max(1.0, box_a[2] - box_a[0])
        ah = max(1.0, box_a[3] - box_a[1])
        bw = max(1.0, box_b[2] - box_b[0])
        bh = max(1.0, box_b[3] - box_b[1])
        width_score = min(aw, bw) / max(aw, bw)
        height_score = min(ah, bh) / max(ah, bh)
        return 0.5 * (width_score + height_score)

    def _predict_bbox(self, memory_item):
        bbox = memory_item["bbox"]
        velocity = memory_item.get("velocity", (0.0, 0.0))
        vx, vy = velocity
        return [bbox[0] + vx, bbox[1] + vy, bbox[2] + vx, bbox[3] + vy]

    def _match_score(self, predicted_bbox, current_bbox):
        iou = self._bbox_iou(predicted_bbox, current_bbox)
        motion = self._bbox_motion_score(predicted_bbox, current_bbox)
        size = self._bbox_size_similarity(predicted_bbox, current_bbox)
        return 0.50 * iou + 0.30 * motion + 0.20 * size

    def _ball_shape_score(self, bbox):
        width = max(1.0, bbox[2] - bbox[0])
        height = max(1.0, bbox[3] - bbox[1])
        ratio = width / height
        return max(0.0, 1.0 - abs(1.0 - ratio))

    def _pick_ball_candidate(self, candidate_boxes, candidate_scores, previous_ball_bbox=None):
        if len(candidate_boxes) == 0:
            return None

        best_idx = 0
        best_score = -1.0
        for idx, (bbox, conf) in enumerate(zip(candidate_boxes, candidate_scores)):
            shape_score = self._ball_shape_score(bbox)
            if previous_ball_bbox is None:
                score = 0.70 * float(conf) + 0.30 * shape_score
            else:
                motion_score = self._bbox_motion_score(previous_ball_bbox, bbox)
                score = 0.55 * float(conf) + 0.20 * shape_score + 0.25 * motion_score

            if score > best_score:
                best_score = score
                best_idx = idx

        return candidate_boxes[best_idx]

    def _recover_stable_id_with_lookback(
        self,
        curr_box,
        stable_memory,
        frame_num,
        used_stable_ids,
        lookback_frames=10,
        match_threshold=0.22,
    ):
        """Try to recover a previous stable ID from recent history before creating a new one."""
        best_stable_id = None
        best_score = -1.0

        for stable_id, mem in stable_memory.items():
            if stable_id in used_stable_ids:
                continue

            age = frame_num - mem["last_frame"]
            if age <= 0 or age > lookback_frames:
                continue

            predicted_bbox = self._predict_bbox(mem)
            score = self._match_score(predicted_bbox, curr_box)
            score -= min(0.10, 0.01 * age)

            if score > best_score:
                best_score = score
                best_stable_id = stable_id

        if best_stable_id is not None and best_score >= match_threshold:
            return best_stable_id
        return None

    def stabilize_player_ids(self, tracks, max_gap=18, match_threshold=0.28):
        """Reduce ID switches by globally matching current players to stable memory."""
        if "players" not in tracks:
            return tracks

        stable_next_id = 1
        stable_memory = {}  # stable_id -> {'bbox': [...], 'last_frame': int, 'velocity': (vx, vy)}
        raw_to_stable_recent = {}  # raw_id -> {'stable_id': int, 'last_frame': int}

        for frame_num, frame_players in enumerate(tracks["players"]):
            remapped_players = {}

            current_items = []
            for raw_id, info in frame_players.items():
                curr_box = info.get("bbox")
                if curr_box is None:
                    continue
                current_items.append((int(raw_id), curr_box, info))

            # Keep only recently seen stable tracks.
            candidate_stable_ids = [
                sid for sid, mem in stable_memory.items()
                if frame_num - mem["last_frame"] <= max_gap
            ]

            assigned_current_indices = set()
            if current_items and candidate_stable_ids:
                score_matrix = np.zeros((len(current_items), len(candidate_stable_ids)), dtype=np.float32)

                for i, (raw_id, curr_box, _) in enumerate(current_items):
                    prev_map = raw_to_stable_recent.get(raw_id)

                    for j, stable_id in enumerate(candidate_stable_ids):
                        mem = stable_memory[stable_id]
                        predicted_bbox = self._predict_bbox(mem)
                        score = self._match_score(predicted_bbox, curr_box)

                        age = frame_num - mem["last_frame"]
                        age_penalty = min(0.20, 0.02 * age)
                        score -= age_penalty

                        # Identity inertia: prefer existing raw->stable mapping if still plausible.
                        if prev_map is not None and prev_map["stable_id"] == stable_id:
                            score += 0.08

                        score_matrix[i, j] = score

                row_ind, col_ind = linear_sum_assignment(-score_matrix)

                for i, j in zip(row_ind, col_ind):
                    score = float(score_matrix[i, j])
                    if score < match_threshold:
                        continue

                    raw_id, curr_box, info = current_items[i]
                    stable_id = candidate_stable_ids[j]
                    mem = stable_memory[stable_id]

                    prev_center = get_center_of_bbox(mem["bbox"])
                    curr_center = get_center_of_bbox(curr_box)
                    vx = float(curr_center[0] - prev_center[0])
                    vy = float(curr_center[1] - prev_center[1])

                    merged_info = dict(info)
                    merged_info["raw_track_id"] = int(raw_id)
                    remapped_players[stable_id] = merged_info

                    stable_memory[stable_id] = {
                        "bbox": curr_box,
                        "last_frame": frame_num,
                        "velocity": (vx, vy),
                    }
                    raw_to_stable_recent[raw_id] = {"stable_id": stable_id, "last_frame": frame_num}

                    assigned_current_indices.add(i)

            # Unmatched detections become new stable IDs.
            for i, (raw_id, curr_box, info) in enumerate(current_items):
                if i in assigned_current_indices:
                    continue

                # Before creating a new ID, try to recover an old stable ID from the last 10 frames.
                stable_id = self._recover_stable_id_with_lookback(
                    curr_box=curr_box,
                    stable_memory=stable_memory,
                    frame_num=frame_num,
                    used_stable_ids=set(remapped_players.keys()),
                    lookback_frames=10,
                    match_threshold=0.22,
                )

                if stable_id is None:
                    stable_id = stable_next_id
                    stable_next_id += 1

                merged_info = dict(info)
                merged_info["raw_track_id"] = int(raw_id)
                remapped_players[stable_id] = merged_info

                prev_mem = stable_memory.get(stable_id)
                if prev_mem is not None:
                    prev_center = get_center_of_bbox(prev_mem["bbox"])
                    curr_center = get_center_of_bbox(curr_box)
                    vx = float(curr_center[0] - prev_center[0])
                    vy = float(curr_center[1] - prev_center[1])
                else:
                    vx, vy = 0.0, 0.0

                stable_memory[stable_id] = {
                    "bbox": curr_box,
                    "last_frame": frame_num,
                    "velocity": (vx, vy),
                }
                raw_to_stable_recent[raw_id] = {"stable_id": stable_id, "last_frame": frame_num}

            tracks["players"][frame_num] = remapped_players

        return tracks

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(
                frames[i:i + self.batch_size],
                conf=self.detection_conf,
                imgsz=self.imgsz,
                verbose=False,
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        previous_ball_bbox = None

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Apply NMS to remove duplicate bounding boxes
            detection_supervision = self._apply_nms_to_detections(detection_supervision, self.nms_threshold)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # FILTER: Remove referees (no longer using them)
            if "referee" in cls_names_inv:
                referee_class_id = cls_names_inv["referee"]
                referee_mask = detection_supervision.class_id != referee_class_id
                detection_supervision = detection_supervision[referee_mask]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                # Skip referees - removed from tracking
                # if cls_id == cls_names_inv['referee']:
                #     tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            ball_class_id = cls_names_inv.get('ball')
            if ball_class_id is not None and detection_supervision.class_id is not None:
                class_ids = detection_supervision.class_id
                confidence = detection_supervision.confidence
                boxes = detection_supervision.xyxy

                ball_mask = class_ids == ball_class_id
                if np.any(ball_mask):
                    ball_boxes = boxes[ball_mask]
                    ball_scores = confidence[ball_mask]
                    selected_ball = self._pick_ball_candidate(ball_boxes, ball_scores, previous_ball_bbox)

                    if selected_ball is not None:
                        selected_ball = selected_ball.tolist()
                        tracks["ball"][frame_num][1] = {"bbox": selected_ball}
                        previous_ball_bbox = selected_ball

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_1_prefix = getattr(self, '_team_1_prefix_counts', None)
        team_2_prefix = getattr(self, '_team_2_prefix_counts', None)
        if team_1_prefix is not None and team_2_prefix is not None and frame_num < len(team_1_prefix):
            team_1_num_frames = int(team_1_prefix[frame_num])
            team_2_num_frames = int(team_2_prefix[frame_num])
        else:
            team_ball_control_till_frame = team_ball_control[:frame_num+1]
            # Get the number of time each team had ball control
            team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
            team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames == 0:
            team_1 = 0.0
            team_2 = 0.0
        else:
            team_1 = team_1_num_frames/total_frames
            team_2 = team_2_num_frames/total_frames

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        if team_ball_control is not None and len(team_ball_control) > 0:
            team_ball_control_array = np.asarray(team_ball_control)
            self._team_1_prefix_counts = np.cumsum(team_ball_control_array == 1)
            self._team_2_prefix_counts = np.cumsum(team_ball_control_array == 2)
        else:
            self._team_1_prefix_counts = None
            self._team_2_prefix_counts = None

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        self._team_1_prefix_counts = None
        self._team_2_prefix_counts = None

        return output_video_frames