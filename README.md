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

<img width="1919" height="912" alt="image" src="https://github.com/user-attachments/assets/2da05cfd-e10c-4984-9ba1-8a35e7a6078d" />

The output video includes:

Player and referee detection with team-based coloring

Annotated player IDs

Camera movement overlay

Player speed and distance metrics

A generated heatmap

Acknowledgment:

I independently designed and implemented this soccer analysis system, including player tracking, camera movement estimation, and calculation of speed and distance metrics. The project was inspired by tutorials from Quick Programming, SuperSimpleDev, Programming with Mosh, and Code in a Jiffy, which helped me understand foundational concepts in programming and computer vision. While I consulted these tutorials for guidance, I adapted and extended the methods to work on my own video clips, add a heatmap generator, and create a unique implementation tailored to soccer footage. I am grateful to these creators for providing resources that enabled me to build this project from scratch.


