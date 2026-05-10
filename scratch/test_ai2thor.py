from ai2thor.controller import Controller
from PIL import Image

try:
    ctrl = Controller(
        scene="FloorPlan1",
        width=224,
        height=224,
        fieldOfView=90
    )
    event = ctrl.step("MoveAhead")
    frame = Image.fromarray(event.frame)
    frame.save("test_frame.png")
    pos = event.metadata["agent"]["position"]
    print(f"SUCCESS — agent at x={pos['x']:.2f}, z={pos['z']:.2f}")
    print(f"Frame size: {frame.size}")
    ctrl.stop()
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}")
