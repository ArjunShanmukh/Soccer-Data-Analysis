from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from heatmap_generator import HeatmapGenerator
def main():
    # Read Video
    video_frames = read_video('C:/Users/sarat/PycharmProjects/PythonProject/football analysis/football_analytics_project.avi')

    # Initialize Tracker
    tracker = Tracker('C:/Users/sarat/PycharmProjects/PythonProject/football analysis/models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='C:/Users/sarat/PycharmProjects/PythonProject/football analysis/stubs/track_stubs.pkl')

    #Get object positions
    tracker.add_position_to_tracks(tracks)

    #camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='C:/Users/sarat/PycharmProjects/PythonProject/football analysis/stubs/camera_movement_stub.pkl')


    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    #View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    # Interpolate Ball Positions
    # tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])  # Comment out this line (line 16)

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = tuple(
                int(x) for x in team_assigner.team_colors[team])

    # Assign ball acquisition
    # player_assigner = PlayerBallAssigner()  # Comment out this line (line 27)
    # team_ball_control = []

    # for frame_num, player_tracks in enumerate(tracks['players']):  # Comment out this line (line 28)
    #     ball_bbox = tracks['ball'][frame_num][1]['bbox']  # Comment out this line (line 29)
    #     assigned_player = player_assigner.assign_ball_to_player(player_tracks, ball_bbox)  # Comment out this line (line 30)
    #
    #     if assigned_player != -1:  # Comment out this line (line 31)
    #         tracks['players'][frame_num][assigned_player]['has_ball'] = True  # Comment out this line (line 32)
    #         team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
    #     else:
    #         team_ball_control.append(team_ball_control[-1])
    # team_ball_control=np.array(team_ball_control)
    # Draw Output
    ##Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, #team_ball_control

                                                    )
    ##Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

      # Heatmap generator
    heatmap_generator = HeatmapGenerator(view_transformer, grid_size=(340, 200))
    heatmap_generator.accumulate_from_tracks(tracks)  # builds accumulator
    output_video_frames = heatmap_generator.draw_heatmap(output_video_frames, alpha=0.5)

    # Build full-frame heatmap
    h_frame, w_frame = output_video_frames[0].shape[:2]
    heatmap_bgr, _ = heatmap_generator.build_heatmap_image(apply_cmap=True)

    # Resize heatmap to full frame
    full_frame_heatmap = cv2.resize(heatmap_bgr, (w_frame, h_frame), interpolation=cv2.INTER_LINEAR)

    # Blend onto a black frame (optional: full alpha 1.0 if you want pure heatmap)
    full_frame_heatmap = cv2.addWeighted(np.zeros_like(full_frame_heatmap, dtype=np.uint8), 0, full_frame_heatmap, 1.0,
                                         0)

    # Append 2-3 seconds of heatmap frames (assuming 30 FPS, 3 seconds → 90 frames)
    fps = 30
    for _ in range(fps * 3):
        output_video_frames.append(full_frame_heatmap)


    # save video
    save_video(output_video_frames, '../output_videos/output_video.avi')

if __name__ == '__main__':

    main()

