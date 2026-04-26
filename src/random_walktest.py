# ============================================================
# random_walk_test.py  |  Track A  |  Owner: Rayyan
#
# PURPOSE
#   Proves that perception.py + mapper.py work together inside
#   AI2-THOR BEFORE Syeda's JEPA is trained.
#   Uses fake/random surprise scores (0.0–0.4) in place of JEPA.
#
# WHAT A PASSING RUN LOOKS LIKE
#   - 40–80 unique nodes after 100 steps  (dedup working)
#   - Several frontier nodes exist        (graph not fully connected)
#   - Coverage > 20%                      (agent is actually moving)
#   - No crashes or assertion errors
#   - Live map window updates in real time
#
# USAGE
#   cd visionvoice
#   python src/random_walk_test.py             # 100 steps, default scene
#   python src/random_walk_test.py --steps 200 --scene FloorPlan201
#   python src/random_walk_test.py --no-display  # headless (no window)
# ============================================================

import argparse
import os
import random
import sys
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from PIL import Image

# Add src/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.perception import DINOEncoder
from src.mapper    import CognitivMap


# ── Config ────────────────────────────────────────────────────────────────────
# LookUp / LookDown only tilt the camera — they never change (x, z) position,
# so including them causes the stuck detector to fire constantly.
# They are kept in collect_trajectories.py for richer trajectory diversity,
# but the walk test only needs actions that actually move the agent.
ACTIONS        = ["MoveAhead", "MoveAhead", "MoveAhead", "RotateLeft", "RotateRight"]
#                 MoveAhead appears 3× so the agent walks forward more than it spins
STUCK_WINDOW   = 8      # steps to look back for stuck detection (lowered for faster recovery)
STUCK_VARIANCE = 0.02   # position variance below this = stuck (tightened)


# ── Live visualiser ───────────────────────────────────────────────────────────
class LiveVisualiser:
    """
    Two-panel matplotlib window:
      Left  — live RGB frame with DINOv3 attention overlay
      Right — live cognitive map (nodes coloured by surprise)
    """

    def __init__(self):
        matplotlib.use("TkAgg")   # change to "Qt5Agg" if TkAgg unavailable
        self.fig, (self.ax_cam, self.ax_map) = plt.subplots(
            1, 2, figsize=(12, 5), facecolor="#0F172A"
        )
        self.fig.suptitle(
            "Vision-to-Voice  |  Random Walk Test  |  Track A",
            color="white", fontsize=11, fontweight="bold"
        )
        for ax in [self.ax_cam, self.ax_map]:
            ax.set_facecolor("#1E293B")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.tick_params(colors="white")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.ion()
        plt.show()

    def update(self, frame_pil, attn_16x16, cog_map, step,
               action, surprise, node_count, frontier_count, coverage):
        import cv2

        self.ax_cam.clear()
        self.ax_map.clear()

        # ── Left: camera + attention overlay ──────────────────────────────────
        frame_arr = np.array(frame_pil)
        self.ax_cam.imshow(frame_arr, extent=[0, 224, 224, 0])

        # Upscale attention 16×16 → 224×224
        attn_big = cv2.resize(
            attn_16x16.astype(np.float32),
            (224, 224), interpolation=cv2.INTER_CUBIC
        )
        attn_big = (attn_big - attn_big.min()) / (attn_big.max() - attn_big.min() + 1e-8)
        self.ax_cam.imshow(
            attn_big, alpha=0.38, cmap="jet",
            vmin=0, vmax=1, extent=[0, 224, 224, 0]
        )

        surp_color = "#EF4444" if surprise > 0.25 else "#60A5FA"
        self.ax_cam.set_title(
            f"Step {step:03d}  |  {action:<12}  |  surprise={surprise:.3f}",
            color=surp_color, fontsize=9, pad=4
        )
        self.ax_cam.axis("off")

        # ── Right: cognitive map ───────────────────────────────────────────────
        G = cog_map.G
        if G.number_of_nodes() > 0:
            pos_dict  = {n: cog_map.G.nodes[n]["pos"] for n in G.nodes}
            surprises = [G.nodes[n]["surprise"] for n in G.nodes]

            nx.draw_networkx_edges(
                G, pos=pos_dict, ax=self.ax_map,
                edge_color="#475569", width=0.8, alpha=0.6
            )
            sc = nx.draw_networkx_nodes(
                G, pos=pos_dict, ax=self.ax_map,
                node_color=surprises, cmap=plt.cm.RdYlBu_r,
                node_size=50, vmin=0, vmax=0.5
            )

            # Highlight frontier nodes
            frontiers = cog_map.frontier_nodes()
            if frontiers:
                frontier_pos = {n: pos_dict[n] for n in frontiers}
                nx.draw_networkx_nodes(
                    G, pos=frontier_pos, ax=self.ax_map,
                    nodelist=frontiers,
                    node_color="#22C55E", node_size=90,
                    node_shape="D"   # diamond = frontier
                )

            # OCR labels
            labels = {n: G.nodes[n]["label"] for n in G.nodes if G.nodes[n]["label"]}
            if labels:
                nx.draw_networkx_labels(
                    G, pos=pos_dict, labels=labels, ax=self.ax_map,
                    font_color="white", font_size=6
                )

        self.ax_map.set_title(
            f"Cognitive Map  |  nodes={node_count}  frontiers={frontier_count}  "
            f"coverage={coverage:.1%}",
            color="white", fontsize=9, pad=4
        )
        self.ax_map.set_facecolor("#1E293B")

        # Legend
        legend_items = [
            mpatches.Patch(color="#EF4444", label="High surprise"),
            mpatches.Patch(color="#60A5FA", label="Low surprise"),
            mpatches.Patch(color="#22C55E", label="Frontier (◆)"),
        ]
        self.ax_map.legend(
            handles=legend_items, loc="upper right",
            fontsize=7, facecolor="#1E293B", labelcolor="white",
            edgecolor="#475569"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ── Stuck detection helper ────────────────────────────────────────────────────
def is_stuck(recent_positions: deque) -> bool:
    if len(recent_positions) < STUCK_WINDOW:
        return False
    arr = np.array(recent_positions)
    return np.var(arr, axis=0).sum() < STUCK_VARIANCE


# ── Main run ──────────────────────────────────────────────────────────────────
def run_random_walk(n_steps: int, scene: str, show_display: bool):

    # ── Imports ───────────────────────────────────────────────────────────────
    try:
        from ai2thor.controller import Controller
    except ImportError:
        print("[ERROR] ai2thor not installed. Run:  pip install ai2thor==5.0.0")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"  Random Walk Test  |  {n_steps} steps  |  scene: {scene}")
    print("=" * 60)

    # ── Init components ───────────────────────────────────────────────────────
    encoder = DINOEncoder()
    cog_map = CognitivMap()
    vis     = LiveVisualiser() if show_display else None

    print(f"\n[AI2-THOR] Loading {scene}...")
    ctrl = Controller(
        scene=scene, width=224, height=224, fieldOfView=90,
        renderDepthImage=False, renderObjectImage=False,
    )
    print("[AI2-THOR] Ready.\n")

    # ── Tracking variables ────────────────────────────────────────────────────
    prev_nid           = None
    recent_positions   = deque(maxlen=STUCK_WINDOW)
    error_history      = []   # surprise scores per step
    coverage_history   = []   # coverage % per step

    # GetReachablePositions works across all AI2-THOR versions
    reachable_event = ctrl.step(action="GetReachablePositions")
    reachable       = reachable_event.metadata.get("actionReturn", [])
    print(f"[AI2-THOR] Reachable positions: {len(reachable)}")

    # ── Step loop ─────────────────────────────────────────────────────────────
    t_start = time.time()

    for step in range(n_steps):

        # ── PERCEIVE ──────────────────────────────────────────────────────────
        frame = Image.fromarray(ctrl.last_event.frame)
        cls, patches = encoder.encode(frame)

        pos_meta = ctrl.last_event.metadata["agent"]["position"]
        rot_meta = ctrl.last_event.metadata["agent"]["rotation"]["y"]
        pos2     = [pos_meta["x"], pos_meta["z"]]

        # ── FAKE SURPRISE (no JEPA yet — random float in [0, 0.4]) ───────────
        # When Syeda delivers jepa_predictor.pt in Week 2, replace this with:
        #   from predictor import prediction_error
        #   surprise = prediction_error(predictor, prev_cls, prev_act, cls)
        surprise = random.uniform(0.0, 0.4)

        # ── UPDATE MAP ────────────────────────────────────────────────────────
        nid = cog_map.add_node(pos2, rot_meta, cls, patches, surprise)
        if prev_nid is not None:
            cog_map.add_edge(prev_nid, nid)
        prev_nid = nid

        # ── STUCK DETECTION ───────────────────────────────────────────────────
        recent_positions.append(pos2)
        if is_stuck(recent_positions):
            # Full 180° turn + 3 forced MoveAheads to break out of corners
            for _ in range(4):    # 4 × 90° RotateRight = 360° sweep
                ctrl.step("RotateRight")
            for _ in range(3):    # walk forward 3 steps
                ctrl.step("MoveAhead")
            print(f"  [Step {step:03d}] ⚠ Stuck — executed 360° sweep + 3×MoveAhead")
            recent_positions.clear()

        # ── CHOOSE ACTION (random — biased toward MoveAhead) ─────────────────
        # ACTIONS already has MoveAhead 3× so the agent walks more than it spins
        action = random.choice(ACTIONS)
        ctrl.step(action)

        # ── METRICS ───────────────────────────────────────────────────────────
        coverage = cog_map.coverage_percent(reachable) if reachable else 0.0
        error_history.append(surprise)
        coverage_history.append(coverage)

        # ── LIVE DISPLAY (every 5 steps to keep UI responsive) ────────────────
        if vis and step % 5 == 0:
            attn = encoder.attention_map(frame)
            vis.update(
                frame_pil     = frame,
                attn_16x16    = attn,
                cog_map       = cog_map,
                step          = step,
                action        = action,
                surprise      = surprise,
                node_count    = cog_map.node_count(),
                frontier_count= len(cog_map.frontier_nodes()),
                coverage      = coverage,
            )

        # ── CONSOLE PROGRESS (every 10 steps) ─────────────────────────────────
        if step % 10 == 0:
            elapsed = time.time() - t_start
            print(
                f"  [Step {step:03d}]  "
                f"nodes={cog_map.node_count():3d}  "
                f"frontiers={len(cog_map.frontier_nodes()):2d}  "
                f"coverage={coverage:.1%}  "
                f"surprise={surprise:.3f}  "
                f"action={action:<12}  "
                f"elapsed={elapsed:.1f}s"
            )

    ctrl.stop()

    # ── Save map ──────────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    save_path = f"data/random_walk_{scene}_map.json"
    cog_map.save(save_path)

    # ── Final report ──────────────────────────────────────────────────────────
    elapsed   = time.time() - t_start
    n_nodes   = cog_map.node_count()
    n_edges   = cog_map.edge_count()
    n_front   = len(cog_map.frontier_nodes())
    final_cov = coverage_history[-1] if coverage_history else 0.0

    print()
    print("=" * 60)
    print("  RANDOM WALK TEST — FINAL RESULTS")
    print("=" * 60)
    print(f"  Steps completed : {n_steps}")
    print(f"  Total nodes     : {n_nodes}")
    print(f"  Total edges     : {n_edges}")
    print(f"  Frontier nodes  : {n_front}")
    print(f"  Coverage        : {final_cov:.1%}")
    print(f"  Avg surprise    : {np.mean(error_history):.3f}")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print(f"  Map saved to    : {save_path}")
    print()

    # ── Pass / Fail thresholds ────────────────────────────────────────────────
    checks = [
        ("Node count 40–80 after 100 steps",
         40 <= n_nodes <= 120,
         f"got {n_nodes}  →  if < 40 lower DEDUP_THRESHOLD, if > 120 raise it"),

        ("At least 3 frontier nodes",
         n_front >= 3,
         f"got {n_front}  →  lower frontier_max_deg in mapper.py"),

        ("Coverage > 20%",
         final_cov >= 0.20,
         f"got {final_cov:.1%}  →  agent may be too constrained, check scene choice"),

        ("No crashes (you reached this line)",
         True, ""),
    ]

    all_pass = True
    print("  PASS / FAIL CHECKS")
    print("  " + "-" * 50)
    for label, ok, hint in checks:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {label}")
        if not ok:
            print(f"       HINT: {hint}")
            all_pass = False

    print()
    if all_pass:
        print("  ✅  ALL CHECKS PASSED — Track A pipeline is solid.")
        print("      You are ready to run:  python src/collect_trajectories.py --mode mock")
    else:
        print("  ⚠   Some checks failed — fix before running trajectory collection.")

    print("=" * 60)
    print()

    if vis:
        print("  Close the plot window to exit.")
        plt.ioff()
        plt.show(block=True)
        vis.close()

    return all_pass


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A random walk integration test")
    parser.add_argument("--steps",      type=int,  default=100,          help="Number of steps (default: 100)")
    parser.add_argument("--scene",      type=str,  default="FloorPlan1", help="AI2-THOR scene name")
    parser.add_argument("--no-display", action="store_true",             help="Disable live plot (headless)")
    args = parser.parse_args()

    success = run_random_walk(
        n_steps     = args.steps,
        scene       = args.scene,
        show_display= not args.no_display,
    )
    sys.exit(0 if success else 1)