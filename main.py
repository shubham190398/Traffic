import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Analysis")

    parser.add_argument(
        "--source_weights_path", required=True, help="Path to source weights file", type=str
    )
    parser.add_argument(
        "--source_video_path", required=True, help="Path to source video file", type=str
    )
    parser.add_argument(
        "--target_video_path", default="output.mp4", help="Path to target video", type=str
    )
    parser.add_argument(
        "--confidence_threshold", default=0.3, help="Confidence threshold", type=float
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold", type=float
    )
