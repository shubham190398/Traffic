import argparse
import numpy as np
import cv2
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

    def process_video(self) -> None:
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)

        for frame in frame_generator:
            processed_frame = self.process_frame(frame=frame)
            cv2.imshow("frame", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(result)

        return self.annotate_frame(frame=frame, detections=detections)
