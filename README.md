Soccer Video Analysis

This project is a soccer video analysis pipeline that detects and tracks players and referees in broadcast footage. It uses a custom-trained YOLOv5n model together with tracking, clustering, and geometric transformations to estimate team assignments, camera movement, and player speed/distance.

Note: The ball is not tracked in this system due to model limitations, and there is no ball interpolation.

Features

Object detection using a fine-tuned YOLOv5n model

Tracking of players and referees with ByteTrack (via Supervision)

Camera movement estimation and adjustment

View transformation from broadcast view to field coordinates

Speed and distance estimation for players

Team color assignment using k-means clustering

Annotated video output with metrics and overlays

Requirements

Python 3.8+

Ultralytics YOLO

Supervision

OpenCV

NumPy

Pandas

scikit-learn

Matplotlib

Install dependencies with:

pip install ultralytics supervision opencv-python numpy pandas scikit-learn matplotlib

Model

The detection model used is a custom-trained YOLOv5n (best.pt), located under models/. Training was performed separately and is not included in this repository, but the provided weights can be used directly for inference.

Usage

Place an input video into the test_videos/ directory.

Update the video path in main.py:

video_frames = read_video('test_videos/my_match_clip.avi')

On the first run with a new video, set read_from_stub=False for both the tracker and camera movement estimator:

tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='stubs/track_stubs.pkl')
camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=False, stub_path='stubs/camera_movement_stub.pkl')

This will run YOLO and optical flow, then cache results into pickle stub files.

On subsequent runs, set read_from_stub=True to reuse cached results and save computation time.

Run the pipeline:

python main.py

The annotated video will be saved under output_videos/output_video.avi.

Notes

Ball tracking is not included. The system only tracks players and referees.

Stub files (track_stubs.pkl, camera_movement_stub.pkl) are tied to a specific input video. Reusing them across different videos will cause mismatches.

The system assumes a broadcast-style camera angle similar to the one used in training. Results may degrade on unconventional views.

Example Output

The output video includes:

Player and referee detection with team-based coloring

Annotated player IDs

Camera movement overlay

Player speed and distance metrics

Acknowledgment:

This file was created after months of watching programming tutorials online, learning how to use Java, CSS, HTML, PHP, etc. from YouTube channels such as Quick Programming, SuperSimpleDev, and Programming with Mosh. In addition, this project required the use of machine learning and deep learning tools like yolo, for which also I watched many tutorials, specifically from the YouTube creator Code in a Jiffy. It is from his analytics tutorial video on tennis that I got the idea to do this project. During the work of this project, I largely worked independently, however, on occasion, I did rewatch some parts of Code in a Jiffy's tennis and soccer analysis tutorials to understand whether I was doing the right thing or not. As such, I would like to thank these creators and Code in a Jiffy especially for putting out such content that helps me and other beginners learn how to code from scratch.


