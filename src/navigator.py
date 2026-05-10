# ============================================================
# navigator.py  |  Integration  |  Week 3
# ============================================================
#
# Pipeline per step:
#   RGB → DINOv2 → CLS → JEPA surprise → Cognitive Map
#   → Frontier Selection → Paper 3 Filter → Escape → Narrate → Execute
#
# Bug history (for the report):
#   v1  threshold=0.25 hardcoded → YOLOE fired 99% of frames
#   v2  x-variance stuck detector → spun RotateRight forever
#   v3  frontier excluded current node on step 0 → quit after 1 step
#   v4  plan_action returned RotateRight at distance=0 → spun forever
#   v5  (this file) — all fixed, see comments inline
#   v6  anti-oscillation: decay-aware frontier scorer, explored marking,
#       ±90° plan_action cone, dispatch cap
# ============================================================

import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import yaml
from PIL import Image

from src.perception       import DINOEncoder
from src.mapper           import CognitivMap
from src.predictor        import load_jepa, prediction_error, action_onehot
from src.scene_classifier import SceneContextMLP
from src.detector         import ConditionalDetector
from src.narrator         import make_narrator

# Dashboard (lazy import — only when enabled)
_dashboard = None

ACTIONS       = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"]
SCENE_ACTIONS = ["move_fast", "stop_wait", "navigate"]


# ■■ 1 — Surprise threshold calibration ■■■■■■■■■■■■■■■■■■■■■■■■■■■■

def calibrate_surprise_threshold(predictor, encoder, controller,
                                  n_warmup=50):
    """
    Run n_warmup random steps and set threshold = mean + 1.0 * std.
    This targets ~16 % of steps exceeding the threshold (1-sigma rule).
    Raw JEPA MSE on 384-dim vectors is typically in [1, 10] — a hardcoded
    0.25 threshold will always fire on nearly every frame.
    """
    import random
    surprises = []
    prev_cls  = None
    prev_act  = None

    print(f"\n[calibrate] {n_warmup} warmup steps …")
    for _ in range(n_warmup):
        frame    = Image.fromarray(controller.last_event.frame)
        cls, _   = encoder.encode(frame)
        if prev_cls is not None:
            surprises.append(
                prediction_error(predictor, prev_cls, prev_act, cls)
            )
        prev_act = random.choice(ACTIONS)
        prev_cls = cls
        controller.step(prev_act)

    if not surprises:
        return 2.0

    mean_s    = float(np.mean(surprises))
    std_s     = float(np.std(surprises))
    # v7: use 1-sigma for ~16% trigger rate; lower clip to allow semantic novelty
    threshold = float(np.clip(mean_s + 1.0 * std_s, 0.05, 20.0))
    print(f"[calibrate] mean={mean_s:.3f}  std={std_s:.3f}  "
          f"threshold={threshold:.3f}")
    
    # Reset controller so exploration starts from spawn, not random warmup end
    controller.reset()
    print(f"[calibrate] Controller reset to spawn.\n")
    print(f"[calibrate] Expected YOLOE trigger rate ≈ 16 % of steps\n")
    return threshold


# ■■ 2 — Wall-aware escape sequencer ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

class EscapeSequencer:
    """
    Wall-aware escape sequencer with backtrack-then-branch logic.
    """
    def __init__(self, narrator=None):
        self.narrator = narrator
        self.state = "idle"
        self.step_idx = 0
        self.sweep_count = 0

    def step(self, last_event, planned_action):
        """
        Args:
            last_event     : controller.last_event after the PREVIOUS action
            planned_action : str — what the planner wants this step
        Returns:
            (action_str, in_escape_bool)
        """
        meta = last_event.metadata
        last_success = meta.get("lastActionSuccess", True)
        last_action = meta.get("lastAction", "")

        if self.state == "idle":
            if last_action == "MoveAhead" and not last_success:
                self.state = "try_left"
                self.step_idx = 0
            else:
                return planned_action, False

        if self.state == "try_left":
            if self.step_idx == 0:
                self.step_idx = 1
                return "RotateLeft", True
            elif self.step_idx == 1:
                self.step_idx = 2
                return "MoveAhead", True
            elif self.step_idx == 2:
                if last_success:
                    self.state = "idle"
                    return planned_action, False
                self.state = "try_back"
                self.step_idx = 0
                
        if self.state == "try_back":
            if self.step_idx == 0:
                self.step_idx = 1
                return "RotateRight", True
            elif self.step_idx == 1:
                self.step_idx = 2
                return "RotateRight", True
            elif self.step_idx == 2:
                self.step_idx = 3
                return "MoveAhead", True
            elif self.step_idx == 3:
                if last_success:
                    if self.narrator:
                        self.narrator.say("Dead end. Returning to last junction.")
                    self.state = "idle"
                    return planned_action, False
                self.state = "sweep_360"
                self.step_idx = 0
                self.sweep_count = 0
                
        if self.state == "sweep_360":
            if self.step_idx == 0:
                self.step_idx = 1
                return "RotateRight", True
            elif self.step_idx == 1:
                self.step_idx = 2
                return "MoveAhead", True
            elif self.step_idx == 2:
                if last_success:
                    self.state = "idle"
                    return planned_action, False
                self.sweep_count += 1
                if self.sweep_count >= 4:
                    if self.narrator:
                        self.narrator.say("Unable to find a clear path. Please assist.")
                    self.state = "idle"
                    return "RotateLeft", False
                self.step_idx = 1
                return "RotateRight", True

        return planned_action, False


# ■■ 3 — Geometric planner ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

def plan_action(controller, cog_map, target_nid):
    """
    Steer toward target_nid using angle arithmetic.

    IMPORTANT: target_nid must never be the agent's current node.
    That filtering is done in the frontier selection step (step 5)
    before this function is ever called. As a result, dx/dz are
    always non-zero and we never hit the distance=0 spin bug.
    """
    if target_nid is None:
        import random
        # Fix 2: Break determinism with a weighted random choice
        return random.choices(
            ["MoveAhead", "RotateLeft", "RotateRight"], 
            weights=[60, 20, 20]
        )[0]

    meta    = controller.last_event.metadata["agent"]
    agent_x = meta["position"]["x"]
    agent_z = meta["position"]["z"]
    rot     = meta["rotation"]["y"]

    target_pos = cog_map.G.nodes[target_nid]["pos"]
    dx = float(target_pos[0]) - agent_x
    dz = float(target_pos[1]) - agent_z

    # Fallback only — should not happen with correct frontier filtering
    if abs(dx) < 0.05 and abs(dz) < 0.05:
        return "MoveAhead"

    angle_to_target = math.degrees(math.atan2(dx, dz)) % 360
    angle_diff = (angle_to_target - rot + 360) % 360

    if 45 < angle_diff <= 180:
        return "RotateRight"
    elif 180 < angle_diff < 315:
        return "RotateLeft"
    else:
        return "MoveAhead"


# ■■ 4 — Scene action bridges ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

def map_to_scene_action(nav_action):
    return "move_fast" if nav_action == "MoveAhead" else "navigate"


def scene_action_to_navigator(scene_action, original):
    if scene_action in ("move_fast", "stop_wait"):
        return "MoveAhead"
    if original in ("RotateLeft", "RotateRight", "LookUp", "LookDown"):
        return original
    return "RotateRight"


# ■■ 5 — Main loop ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

def run_agent(controller, encoder, predictor, scene_clf, detector,
              narrator, cog_map, n_steps=300, surprise_threshold=None,
              collector=None, saliency_engine=None,
              enable_dashboard=False):
    """
    Main navigation loop — the heart of Vision-to-Voice.

    Step-by-step pipeline:
        1. PERCEIVE    — DINOv2 encode current frame
        2. SURPRISE    — JEPA prediction error vs previous step
        3. MAP         — add node, connect to previous node
        4. DETECT      — YOLOE + OCR if surprise > threshold
        5. FRONTIER    — pick target that is NOT the current node
        6. FILTER      — Paper 3 scene MLP appropriateness check
        7. ESCAPE      — EscapeSequencer reads lastActionSuccess
        8. NARRATE     — voice update every 5 steps
        9. EXECUTE     — send action to AI2-THOR

    Key fix in step 5 (v5):
        frontier_nodes(current_nid=nid) uses mapper's degree-aware logic:
          - degree=0 (step 0, only 1 node): current node IS returned as a
            frontier so the agent has a target and takes MoveAhead
          - degree>=1 (step 1+): current node excluded so plan_action
            always has a genuinely distant target with dx/dz > 0
    """
    if surprise_threshold is None:
        surprise_threshold = calibrate_surprise_threshold(
            predictor, encoder, controller, n_warmup=50
        )

    prev_cls  = None
    prev_act  = None
    prev_nid  = None
    error_log = []
    escaper   = EscapeSequencer(narrator)
    paper3_override_count = 0
    yoloe_trigger_count   = 0
    _dispatched_count     = {}   # anti-oscillation: cap per-node targeting
    _ocr_text_latest      = ""   # latest OCR text for dashboard display

    # ── Dashboard init ───────────────────────────────────────
    global _dashboard
    if enable_dashboard:
        import matplotlib.pyplot as plt
        from demo.dashboard import init_dashboard, update as dash_update
        init_dashboard()
        _dashboard = dash_update
        print("[dashboard] Live dashboard enabled")
    else:
        _dashboard = None

    print(f"\n{'='*60}")
    print(f"  Vision-to-Voice  |  {n_steps} steps  |  "
          f"threshold={surprise_threshold:.3f}")
    print(f"{'='*60}\n")
    narrator.say("Vision to Voice system online. Beginning exploration.")

    for step in range(n_steps):

        # ── 1. PERCEIVE ──────────────────────────────────────────
        frame     = Image.fromarray(controller.last_event.frame)
        cls, ptch = encoder.encode(frame)
        meta      = controller.last_event.metadata["agent"]
        pos2      = [meta["position"]["x"], meta["position"]["z"]]
        rot       = meta["rotation"]["y"]

        # ── 2. SURPRISE ──────────────────────────────────────────
        surprise = (prediction_error(predictor, prev_cls, prev_act, cls)
                    if prev_cls is not None else 0.0)
        error_log.append(surprise)

        # ── 3. MAP ───────────────────────────────────────────────
        nid = cog_map.add_node(pos2, rot, cls, ptch, surprise)
        if prev_nid is not None:
            cog_map.add_edge(prev_nid, nid)

        # ── 4. CONDITIONAL YOLOE + OCR ───────────────────────────
        # v7: Only trigger YOLOE if the last action succeeded. 
        # Surprise on failure (collision) is expected physics, not novelty.
        last_success = controller.last_event.metadata.get("lastActionSuccess", True)
        
        # v10: Diagnostic for scale mismatch
        if prev_cls is not None:
            with torch.no_grad():
                z_pred_norm = torch.norm(predictor(prev_cls.unsqueeze(0), action_onehot(prev_act).unsqueeze(0))).item()
                z_act_norm  = torch.norm(cls).item()
                if step % 20 == 0:
                    print(f"[debug] pred_norm={z_pred_norm:.3f}  act_norm={z_act_norm:.3f}")

        # v12: Diagnostic — expose surprise vs threshold on every step for first 10
        if step < 10:
            print(f"[yoloe-check] step={step} surprise={surprise:.4f} threshold={surprise_threshold:.4f} check={surprise > surprise_threshold} last_success={last_success}")

        if surprise > surprise_threshold and last_success:
            yoloe_trigger_count += 1
            objects, ocr_text = detector.run(frame)
            if ocr_text:
                cog_map.tag_label(nid, ocr_text)
                narrator.say(f"Sign detected: {ocr_text}")
                _ocr_text_latest = ocr_text
            if objects:
                narrator.say(f"I can see: {', '.join(objects[:3])}")

        # ── 5. FRONTIER SELECTION ─────────────────────────────────
        # Hybrid approach: Situation A vs Situation B
        if cog_map.G.degree(nid) < cog_map.MAX_DEGREE:
            # Situation A: Agent is AT a frontier. Step outward to discover new space.
            target = None
        else:
            # Situation B: Agent is deep in explored territory. Navigate to a distant frontier.
            all_frontiers = cog_map.frontier_nodes()
            frontiers = [f for f in all_frontiers if f != nid and f != prev_nid]

            if frontiers:
                target = max(
                    frontiers,
                    key=lambda n: cog_map.score_frontier(
                        n,
                        current_pos=pos2,
                        dispatched_count=_dispatched_count.get(n, 0),
                    ),
                )
                _dispatched_count[target] = _dispatched_count.get(target, 0) + 1
            else:
                target = None



        # ── 6. PAPER 3 FILTER ────────────────────────────────────
        raw_action   = plan_action(controller, cog_map, target)
        scene_action = map_to_scene_action(raw_action)
        filtered     = scene_clf.filter(scene_action, cls, threshold=0.4)

        if filtered != scene_action:
            action = scene_action_to_navigator(filtered, raw_action)
            paper3_override_count += 1
            print(f"[paper3] step={step} proposed={scene_action} filtered={filtered} scores={scene_clf(cls).squeeze().tolist()}")
        else:
            action = raw_action

        # ── 7. ESCAPE SEQUENCER ──────────────────────────────────
        action, in_escape = escaper.step(controller.last_event, action)
        if in_escape:
            print(f"  [Escape] step={step}  action={action}  "
                  f"state={escaper.state}")

        # ── 8. NARRATE ───────────────────────────────────────────
        if step % 5 == 0:
            narrator.say(
                f"Step {step}. {action}. "
                f"{cog_map.node_count()} locations mapped. "
                f"Surprise {'high' if surprise > surprise_threshold else 'low'}."
            )

        # ── 8.2. LIVE DASHBOARD ──────────────────────────────────
        if _dashboard is not None and step % 5 == 0:
            import matplotlib.pyplot as plt
            try:
                # Attention map from DINOv3
                _attn = encoder.attention_map(frame)

                # SmoothGrad saliency (fast: n_samples=10)
                _ftensor = encoder._preprocess(frame)
                if saliency_engine is not None:
                    _smap = saliency_engine.smoothgrad_map(_ftensor, n_samples=10)
                else:
                    _smap = None

                # Paper 3 action scores
                _ascores = scene_clf.get_scores(cls)

                _dashboard(
                    frame=frame,
                    attn_map=_attn,
                    smap=_smap,
                    graph=cog_map.G,
                    error_history=error_log,
                    action_scores=_ascores,
                    current_node_id=nid,
                    ocr_text=_ocr_text_latest,
                    surprise_threshold=surprise_threshold,
                )
                plt.pause(0.01)
                _ocr_text_latest = ""  # clear after display
            except Exception as e:
                print(f"[dashboard] update error: {e}")

        # ── 8.5. ATTNLRP COLLECTION ──────────────────────────────
        if collector is not None and saliency_engine is not None:
            frame_tensor = encoder._preprocess(frame)
            attn_map = saliency_engine.get_map(frame_tensor, method="attnlrp")
            current_scene = controller.last_event.metadata.get("sceneName", "unknown")
            collector.record(frame, attn_map, action, surprise, current_scene, nid)

        # ── 9. EXECUTE ───────────────────────────────────────────
        # v11: diagnostic action logging
        if step % 5 == 0:
            print(f"[debug] step={step} target={target} raw={raw_action} final={action} last_success={controller.last_event.metadata.get('lastActionSuccess', True)}")
            
        controller.step(action)
        prev_nid = nid
        prev_cls = cls
        prev_act = action

    # ── Summary ──────────────────────────────────────────────────
    trigger_pct = 100 * yoloe_trigger_count / max(len(error_log), 1)
    print(f"\n{'='*60}")
    print(f"  Navigation Complete — {len(error_log)} steps executed")
    print(f"  Nodes mapped     : {cog_map.node_count()}")
    print(f"  Edges built      : {cog_map.edge_count()}")
    print(f"  YOLOE triggers   : {yoloe_trigger_count} ({trigger_pct:.1f}%)")
    print(f"  Paper 3 overrides: {paper3_override_count}")
    print(f"{'='*60}\n")
    return cog_map, error_log


# ■■ 6 — Entry point ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "config.yaml"
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("  Vision-to-Voice Navigator — Integration Test")
    print("=" * 60)

    print("\n[1/6] Starting AI2-THOR …")
    from ai2thor.controller import Controller
    controller = Controller(
        scene       = "FloorPlan4",
        width       = cfg["ai2thor"]["width"],
        height      = cfg["ai2thor"]["height"],
        fieldOfView = cfg["ai2thor"]["fov"],
    )

    print("\n[2/6] Loading DINOv2 encoder …")
    encoder = DINOEncoder()

    print("\n[3/6] Loading JEPA world model …")
    predictor = load_jepa(
        model_path = cfg["jepa"]["model_path"],
        z_dim      = cfg["dino"]["cls_dim"],
        act_dim    = 5,
        hidden     = cfg["jepa"]["hidden_dim"],
    )
    predictor.eval()

    print("\n[4/6] Loading Scene Context MLP …")
    scene_clf = SceneContextMLP(input_dim=cfg["dino"]["cls_dim"])
    scene_clf.load_state_dict(
        torch.load(cfg["scene_mlp"]["model_path"],
                   map_location="cpu", weights_only=True)
    )
    scene_clf.eval()

    print("\n[5/6] Loading YOLOE + EasyOCR …")
    detector = ConditionalDetector()

    print("\n[6/6] Initializing narrator + saliency …")
    narrator = make_narrator(enabled=True)

    from src.saliency import SaliencyEngine
    saliency_engine = SaliencyEngine(encoder.model)

    print("\n" + "=" * 60)
    print("  Starting 200-step exploration run …")
    print("  (First 50 steps = calibration warmup)")
    print("=" * 60)

    cog_map = CognitivMap()
    cog_map, error_log = run_agent(
        controller         = controller,
        encoder            = encoder,
        predictor          = predictor,
        scene_clf          = scene_clf,
        detector           = detector,
        narrator           = narrator,
        cog_map            = cog_map,
        n_steps            = 200,
        surprise_threshold = None,
        saliency_engine    = saliency_engine,
        enable_dashboard   = True,
    )

    avg_s   = float(np.mean(error_log)) if error_log else 0.0
    max_s   = float(np.max(error_log))  if error_log else 0.0
    labeled = cog_map.labeled_nodes()

    print("\n" + "=" * 60)
    print("  FINAL INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"  Total nodes mapped    : {cog_map.node_count()}")
    print(f"  Total edges           : {cog_map.edge_count()}")
    print(f"  Average surprise      : {avg_s:.4f}")
    print(f"  Max surprise          : {max_s:.4f}")
    print(f"  Named nodes (OCR)     : {len(labeled)}")
    for nid, label in labeled:
        print(f"    Node {nid}: \"{label}\"")
    print(f"  Steps executed        : {len(error_log)}")
    print("=" * 60)

    node_ok = cog_map.node_count() > 50
    print(f"  {'[OK]' if node_ok else '[!] '} Node count "
          f"{'> 50 — healthy exploration' if node_ok else '<= 50 — agent may be stuck'}")

    narrator.say("Navigation test complete. Shutting down.")
    if hasattr(narrator, "shutdown"):
        narrator.shutdown()
    controller.stop()
    print("\n  Controller stopped. Test complete.\n")