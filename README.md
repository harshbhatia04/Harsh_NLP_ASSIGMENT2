# Human Pose Estimation using YOLO

This project uses YOLO26 to detect body keypoints in a video, including shoulders, elbows, wrists, hips, knees and ankles. The points are connected to show a moving skeleton.

It creates two outputs: a white skeleton over the original video, and a white skeleton on a black background.

## Project files

| File | Description |
| --- | --- |
| `human_skeleton.py` | Python source code |
| `video run.mp4` | Original test video |
| `video run.avi` | Skeleton over the original video |
| `white_skeleton.mp4` | White skeleton on a black background |
| `README.md` | Setup and project details |

## Installation

Install Python, then run:

```bash
python -m pip install ultralytics==8.4.193 opencv-python numpy imageio-ffmpeg lap
```

## How to run

Open a terminal in this folder and run:

```bash
python human_skeleton.py
```

The script processes the complete video and replaces both output videos. It preserves the frame rate and resolution; odd dimensions receive one black edge pixel. The outputs are silent.

To use your own recording, replace `video run.mp4` with your video using the same filename, then run the script again. The current clip is only a sample for testing.

## Videos

- [Original video](video%20run.mp4)
- [Pose overlay](video%20run.avi)
- [White skeleton on black](white_skeleton.mp4)

## Model used

The pretrained Ultralytics `yolo26n-pose.pt` model detects 17 body keypoints. ByteTrack keeps track of people between frames. Joints with confidence below 0.35 are skipped to reduce incorrect connections.

The model downloads on the first run and is stored in `.cache/yolo-pose` inside your user folder, outside this project. No training is required. Hidden joints and fast movements can cause missing or inaccurate points.

The current sample shows two people running. It is 24.32 seconds long: 608 frames at 25 FPS, with a resolution of 1920 x 1080. The supplied `6012e445c9.mp4` is saved here as `video run.mp4` to keep the same project filenames. Both outputs are generated from this video.

[Ultralytics pose documentation](https://docs.ultralytics.com/tasks/pose/)
