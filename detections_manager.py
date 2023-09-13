from typing import Dict, List
import supervision as sv
import numpy as np


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}

    def update(self, detections: sv.Detections, detections_zones_in: List[sv.Detections]) -> sv.Detections:
        for zone_in_id, detections_zone_in in enumerate(detections_zones_in):
            for tracker_id in detections_zone_in.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        detections.class_id = np.vectorize(lambda x: self.tracker_id_to_zone_id.get(x, -1))(detections.tracker_id)
        detections = detections[detections.class_id != -1]
        return detections
