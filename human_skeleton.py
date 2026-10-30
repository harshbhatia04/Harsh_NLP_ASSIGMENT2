"""Create a pose overlay and a white skeleton video using YOLO26."""

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO


FOLDER = Path(__file__).resolve().parent
INPUT_VIDEO = FOLDER / "video run.mp4"
POSE_VIDEO = FOLDER / "video run.avi"
SKELETON_VIDEO = FOLDER / "white_skeleton.mp4"

# Connections between the 17 COCO body keypoints.
CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (3, 5), (4, 6), (5, 6),                 # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


def draw_skeleton(image, points, scores):
    height, width = image.shape[:2]
    visible = (
        (scores >= 0.35)
        & np.isfinite(points).all(axis=1)
        & (points[:, 0] > 0) & (points[:, 0] < width)
        & (points[:, 1] > 0) & (points[:, 1] < height)
    )
    thickness = max(2, round(min(width, height) / 240))
    radius = max(3, round(thickness * 1.5))
    white = (255, 255, 255)

    for start, end in CONNECTIONS:
        if visible[start] and visible[end]:
            a = tuple(np.rint(points[start]).astype(int))
            b = tuple(np.rint(points[end]).astype(int))
            cv2.line(image, a, b, white, thickness, cv2.LINE_AA)

    for point in points[visible]:
        centre = tuple(np.rint(point).astype(int))
        cv2.circle(image, centre, radius, white, -1, cv2.LINE_AA)


def main():
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    overlay_writer = None
    skeleton_writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot read the input video: {INPUT_VIDEO}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot read the video frame rate.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width = frame.shape[:2]
        size = (width + width % 2, height + height % 2)

        # Store downloaded weights outside the project folder.
        model_path = Path.home() / ".cache" / "yolo-pose" / "yolo26n-pose.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(model_path))

        overlay_writer = cv2.VideoWriter(
            str(POSE_VIDEO), cv2.VideoWriter_fourcc(*"XVID"), fps, size,
        )
        if not overlay_writer.isOpened():
            raise RuntimeError("Cannot create the pose overlay video.")
        skeleton_writer = imageio_ffmpeg.write_frames(
            str(SKELETON_VIDEO), size, fps=fps, codec="libx264",
            pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            quality=None, macro_block_size=1,
            output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart"],
        )
        skeleton_writer.send(None)

        count = 0
        while ok:
            # Keep the person's tracking ID between consecutive frames.
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=0.25, imgsz=640, device="cpu", verbose=False,
            )[0]
            black = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            overlay = cv2.copyMakeBorder(
                frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT,
            )
            if result.keypoints is not None and result.keypoints.conf is not None:
                points = result.keypoints.xy.cpu().numpy()
                scores = result.keypoints.conf.cpu().numpy()
                for person, confidence in zip(points, scores):
                    draw_skeleton(black, person, confidence)
                    draw_skeleton(overlay, person, confidence)

            # Include blank frames too, so the video keeps its original timing.
            overlay_writer.write(overlay)
            skeleton_writer.send(black)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count}/{total} frames", flush=True)
            ok, frame = capture.read()

        if total > 0 and count != total:
            raise RuntimeError(f"Only {count} of {total} frames could be read.")
    finally:
        capture.release()
        if overlay_writer is not None:
            overlay_writer.release()
        if skeleton_writer is not None:
            skeleton_writer.close()

    print(f"Saved {POSE_VIDEO.name} and {SKELETON_VIDEO.name}")


if __name__ == "__main__":
    main()
