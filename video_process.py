import numpy as np
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
from typing import List, Tuple
from detections_manager import DetectionsManager


ZONE_IN_POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
]

COLORS = sv.ColorPalette.default()


def initiate_polygons(
        polygons: List[np.ndarray],
        frame_resolution: Tuple[int, int],
        triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(polygon=polygon, frame_resolution_wh=frame_resolution, triggering_position=triggering_position)
        for polygon in polygons
    ]


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
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, thickness=2, trace_length=100)
        self.tracker = sv.ByteTrack()
        self.video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        self.zones_in = initiate_polygons(ZONE_IN_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER)
        self.zones_out = initiate_polygons(ZONE_OUT_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER)
        self.detections_manager = DetectionsManager()

    def process_video(self) -> None:
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)

        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                processed_frame = self.process_frame(frame=frame)
                sink.write_frame(processed_frame)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)

        detections_zones_in = []
        detections_zones_out = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_zone_in = detections[zone_in.trigger(detections=detections)]
            detections_zones_in.append(detections_zone_in)
            detections_zone_out = detections[zone_out.trigger(detections=detections)]
            detections_zones_out.append(detections_zone_out)

        detections = self.detections_manager.update(
            detections=detections,
            detections_zones_in=detections_zones_in,
            detections_zones_out=detections_zones_out,
        )
        return self.annotate_frame(frame=frame, detections=detections)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=zone_in.polygon, color=COLORS.colors[i])
            annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=zone_out.polygon, color=COLORS.colors[i])

        labels = [
            f"{tracker_id}"
            for tracker_id in detections.tracker_id
        ]
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)

            if zone_out_id in self.detections_manager.recorded_paths:
                paths = self.detections_manager.recorded_paths[zone_out_id]

                for i, zone_in_id in enumerate(paths):
                    count = len(paths[zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=f"{count}",
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id]
                    )

        return annotated_frame
