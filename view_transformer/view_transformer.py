import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        # self.pixel_vertices = np.array([[110, 1035], 
        #                        [265, 275], 
        #                     #    [910, 260], 
        #                     #    [1640, 915]])
        #                        [1100, 260], 
        #                        [1920, 1000]])
        self.pixel_vertices = np.array([[150, 1050], 
                               [400, 300], 
                               [1920, 250], 
                               [1900, 1000]])
        
        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self,point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None
        # reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        # tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        # result = tranform_point.reshape(-1,2)
        try:
            reshaped_point = point.reshape(-1,1,2).astype(np.float32)
            tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
            result = tranform_point.reshape(-1,2)
            
            # Check for invalid values (NaN or Inf)
            if np.isnan(result).any() or np.isinf(result).any():
                print(f"[WARNING] Invalid transformation result: {result} for point {point}")
                return None
            
            return result
        except Exception as e:
            print(f"[ERROR] Transform failed for point {point}: {e}")
            return None

    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed