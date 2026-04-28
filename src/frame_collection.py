from ai2thor.controller import Controller
from PIL import Image
import random, json

ctrl = Controller(scene="FloorPlan22", width=224, height=224, fieldOfView=90)
ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight"]

frames = []
for i in range(300):
    event = ctrl.step(random.choice(ACTIONS))
    frame = Image.fromarray(event.frame)
    frame.save(f"data/cafeteria(kitchen)/frame_{i:04d}.png")
    pos = event.metadata["agent"]["position"]
    frames.append({
        "frame_id": f"frame_{i:04d}.png",
        "pos": [pos["x"], pos["z"]]
    })

ctrl.stop()