import os
import glob
import random
import math
import numpy as np
from PIL import Image


class MockEvent:
    def __init__(self, frame, scene, action, success, position, rotation):
        self.frame = frame
        self.metadata = {
            "agent": {
                "position": {"x": position[0], "z": position[1]},
                "rotation": {"y": rotation}
            },
            "sceneName": scene,
            "lastActionSuccess": success,
            "lastAction": action
        }


class MockController:
    """
    A mock AI2-THOR controller that serves campus photos and simulates
    realistic incremental movement (not random teleportation).

    Movement model:
        MoveAhead  → advance 0.25m in facing direction
        RotateLeft → yaw -= 90°
        RotateRight→ yaw += 90°
        LookUp/Down→ no position change (camera tilt only)

    This produces plausible navigation trajectories for the cognitive map.
    """

    STEP_SIZE = 0.25  # metres per MoveAhead
    ROTATE_DEG = 90   # degrees per rotation

    def __init__(self, scene="FloorPlan1", width=224, height=224, fieldOfView=90, **kwargs):
        self.scene = scene
        self.width = width
        self.height = height

        # Agent state
        self._pos = [0.0, 0.0]  # (x, z)
        self._rot = 0.0         # yaw in degrees

        # Load all campus photos
        photo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "campus_photos")
        self.photo_paths = sorted(glob.glob(os.path.join(photo_dir, "*.*")))
        self.photo_paths = [p for p in self.photo_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not self.photo_paths:
            self.photo_paths = [None]  # fallback to random noise

        # Shuffle for variety but keep repeatable within a scene
        random.shuffle(self.photo_paths)
        self._photo_idx = 0

        self.last_event = self._generate_event("Initialize", True)

    def _next_photo(self):
        """Cycle through photos, re-shuffling when exhausted."""
        path = self.photo_paths[self._photo_idx % len(self.photo_paths)]
        self._photo_idx += 1
        # Re-shuffle after one full pass to keep variety
        if self._photo_idx % len(self.photo_paths) == 0:
            random.shuffle(self.photo_paths)
        return path

    def _generate_event(self, action, success):
        path = self._next_photo()
        if path:
            img = Image.open(path).convert("RGB").resize((self.width, self.height))
            frame = np.array(img)
        else:
            frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

        return MockEvent(frame, self.scene, action, success,
                         position=list(self._pos), rotation=self._rot)

    def step(self, action, **kwargs):
        # Simulate movement
        if action == "MoveAhead":
            # 10% chance of failure (wall collision)
            success = random.random() > 0.10
            if success:
                rad = math.radians(self._rot)
                self._pos[0] += self.STEP_SIZE * math.sin(rad)
                self._pos[1] += self.STEP_SIZE * math.cos(rad)
        elif action == "RotateLeft":
            self._rot = (self._rot - self.ROTATE_DEG) % 360
            success = True
        elif action == "RotateRight":
            self._rot = (self._rot + self.ROTATE_DEG) % 360
            success = True
        elif action in ("LookUp", "LookDown"):
            success = True
        else:
            success = True

        self.last_event = self._generate_event(action, success)
        return self.last_event

    def reset(self, scene=None):
        if scene:
            self.scene = scene
        self._pos = [0.0, 0.0]
        self._rot = 0.0
        self._photo_idx = 0
        random.shuffle(self.photo_paths)
        self.last_event = self._generate_event("Initialize", True)

    def stop(self):
        pass
