import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO


class VideoProcessor:
    def __init__(
            self,
            source_weights_path: str,
            source_video_path: str,
            target_video_path: str = None,
            confidence_threshold: float = 0.3,
            iou_threshold: float = 0.7,
    ) -> None:
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(source_weights_path)
