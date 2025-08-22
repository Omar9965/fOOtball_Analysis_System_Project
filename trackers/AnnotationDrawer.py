import numpy as np
import cv2
from utils import get_center_of_bbox, get_bbox_width


class AnnotationDrawer:

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        h, w, _ = frame.shape

        # Ensure numpy array
        team_ball_control = np.array(team_ball_control)

        # Handle case where frame_num >= len(team_ball_control)
        if frame_num >= len(team_ball_control):
            return frame

        # Rectangle & text scaling
        if w <= 640: 
            rect_x1, rect_y1, rect_x2, rect_y2 = 200, int(h * 0.8), 550, int(h * 0.95)
            font_scale = 0.5
            line_height = 25
        elif w <= 1280: 
            rect_x1, rect_y1, rect_x2, rect_y2 = 700, int(h * 0.85), 1160, int(h * 0.97)
            font_scale = 1
            line_height = 35
        else:  
            rect_x1, rect_y1, rect_x2, rect_y2 = int(w*0.55), int(h*0.85), int(w*0.9), int(h*0.97)
            font_scale = max(0.8, w / 1920)  
            line_height = int(40 * font_scale)

        # Semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Ball control calculation
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)

        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames == 0:
            team_1, team_2 = 0, 0
        else:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames

        # Text positions (line height adapts)
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.1f}%",
                    (rect_x1 + 10, rect_y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.1f}%",
                    (rect_x1 + 10, rect_y1 + 30 + line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

        return frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames