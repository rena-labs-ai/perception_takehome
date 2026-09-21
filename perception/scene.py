"""Read one extracted scene: data/<scene>/{rgb,depth}/NNNNNN.png + calib.json (+ boxes.json on dev scenes)."""

import json
from pathlib import Path

import cv2
import numpy as np


class Scene:
    def __init__(self, root, calib_path=None):
        self.root = Path(root)
        self.calib = json.loads(
            Path(calib_path or self.root / "calib.json").read_text()
        )
        self.frames = self.calib["frames"]
        boxes = self.root / "boxes.json"
        self.boxes = json.loads(boxes.read_text()) if boxes.exists() else None

    @property
    def name(self):
        return self.root.name

    @property
    def color(self):
        return self.calib["color"]

    @property
    def depth_cam(self):
        return self.calib["depth"]

    @property
    def T_color_depth(self):
        return np.array(self.calib["depth"]["T_color_depth"], dtype=np.float64)

    def __len__(self):
        return len(self.frames)

    def pose(self, i):
        """Camera-to-world transform of frame i (4x4)."""
        return np.array(self.frames[i]["T_world_camera"], dtype=np.float64)

    def rgb(self, i):
        return cv2.cvtColor(
            cv2.imread(str(self.root / "rgb" / f"{i:06d}.png")), cv2.COLOR_BGR2RGB
        )

    def depth_raw(self, i):
        return cv2.imread(
            str(self.root / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED
        )

    def depth_m(self, i):
        return self.depth_raw(i).astype(np.float32) * self.depth_cam["scale_m"]
