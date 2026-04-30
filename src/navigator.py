# ============================================================
# navigator.py  |  Integration  |  Day 5 — connects all tracks
# ============================================================
#
# This is the MAIN LOOP that ties together:
#   Track A (Rayyan) : perception.py (DINOEncoder), mapper.py (CognitivMap)
#   Track B (Syeda)  : predictor.py  (JEPAPredictor), scene_classifier.py (SceneContextMLP)
#   Track C (Riya)   : detector.py   (ConditionalDetector), narrator.py (Narrator)
#
# Full pipeline per step:
#   RGB frame → DINOv2 (frozen) → CLS token → JEPA surprise
#   → Cognitive Map → Frontier Selection → Paper 3 Filter → Action → AI2-THOR
#
# Paper references:
#   Paper 1 (Mirowski et al. 2017) — JEPA surprise drives frontier exploration
#   Paper 2 (Gupta et al. 2017)   — Topological cognitive map, not metric grid
#   Paper 3 (Epstein et al. 2019) — Learned action-place compatibility filter
#   Novel contribution            — Conditional YOLOE trigger via JEPA surprise
# ============================================================

import sys
import os
import numpy as np
import torch
import yaml
from PIL import Image
from collections import deque

# ── Track A imports (Rayyan) ─────────────────────────────────────
from src.perception import DINOEncoder
from src.mapper import CognitivMap              # NOTE: class is CognitivMap (no 'e')

# ── Track B imports (Syeda) ──────────────────────────────────────
from src.predictor import load_jepa, prediction_error
from src.scene_classifier import SceneContextMLP

# ── Track C imports (Riya) ───────────────────────────────────────
from src.detector import ConditionalDetector
from src.narrator import Narrator, make_narrator


# ■■ Constants ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

# AI2-THOR action vocabulary — these are the ONLY valid actions
ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"]

# Scene classifier action vocabulary — Paper 3 (Epstein et al. 2019)
SCENE_ACTIONS = ["move_fast", "stop_wait", "navigate"]


# ■■ Helper: Geometric Planner ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

def plan_action(controller, cog_map, target_nid):
    """
    Simple geometric planner — decides which AI2-THOR action to take
    to move toward a target node in the cognitive map.

    Strategy:
        1. Get agent's current (x, z) position from controller metadata
        2. Get target node's (x, z) position from the cognitive map graph
        3. Compute dx and dz between agent and target
        4. If |dx| > |dz|: turn toward the target (RotateRight/RotateLeft)
        5. Otherwise: walk forward (MoveAhead)

    This is intentionally simple — the cognitive map handles high-level
    planning (which node to visit), this function just handles low-level
    steering toward that node.

    Paper 2 (Gupta et al. 2017): topological navigation decomposes into
    graph-level planning + local motor control. This is the motor control.

    Args:
        controller  : ai2thor.controller.Controller — live simulation handle
        cog_map     : CognitivMap — the topological cognitive map
        target_nid  : int or None — target node ID in cog_map.G

    Returns:
        str — one of ACTIONS ("MoveAhead", "RotateLeft", "RotateRight")
    """
    # If no target, just walk forward (exploration default)
    if target_nid is None:
        return "MoveAhead"

    # Current agent position from AI2-THOR metadata
    pos = controller.last_event.metadata["agent"]["position"]
    agent_x, agent_z = pos["x"], pos["z"]

    # Target node position from cognitive map graph
    target_pos = cog_map.G.nodes[target_nid]["pos"]
    target_x, target_z = target_pos[0], target_pos[1]

    # Vector from agent to target
    dx = target_x - agent_x
    dz = target_z - agent_z

    # If lateral offset dominates, rotate to face the target
    if abs(dx) > abs(dz):
        return "RotateRight" if dx > 0 else "RotateLeft"

    # Otherwise, move forward toward the target
    return "MoveAhead"


# ■■ Helper: Navigator ↔ Scene Classifier Action Mapping ■■■■■■■■■

def map_to_scene_action(navigator_action):
    """
    Convert AI2-THOR navigator action to scene classifier action vocabulary.

    Paper 3 (Epstein et al. 2019) uses a different action vocabulary than
    AI2-THOR. This function bridges the two:

        MoveAhead              → "move_fast"   (linear motion)
        RotateLeft/RotateRight → "navigate"    (turning/orienting)
        LookUp/LookDown       → "navigate"    (camera adjustment)

    Args:
        navigator_action: str — one of ACTIONS

    Returns:
        str — one of SCENE_ACTIONS ("move_fast", "stop_wait", "navigate")
    """
    if navigator_action == "MoveAhead":
        return "move_fast"
    # All rotation/look actions map to "navigate"
    # (RotateLeft, RotateRight, LookUp, LookDown)
    return "navigate"


def scene_action_to_navigator(scene_action, original_navigator_action):
    """
    Reverse-map a scene classifier action back to an AI2-THOR action.

    When Paper 3 overrides the proposed action, we need to convert the
    scene classifier's chosen action back into a valid AI2-THOR action.

    Mapping:
        "move_fast" → "MoveAhead"
        "stop_wait" → "MoveAhead"  (we don't have a true stop, move gently)
        "navigate"  → keep original if it was a rotation, else "RotateRight"

    Args:
        scene_action              : str — the action Paper 3 chose
        original_navigator_action : str — what we originally planned

    Returns:
        str — a valid AI2-THOR action from ACTIONS
    """
    if scene_action == "move_fast":
        return "MoveAhead"
    elif scene_action == "stop_wait":
        # No literal "stop" in AI2-THOR — default to moving ahead slowly
        # (the agent will still explore, just flagged as cautious context)
        return "MoveAhead"
    elif scene_action == "navigate":
        # Keep the original rotation if it was already a rotation action
        if original_navigator_action in ("RotateLeft", "RotateRight",
                                          "LookUp", "LookDown"):
            return original_navigator_action
        # Otherwise default to a rotation to reorient
        return "RotateRight"
    # Fallback — should never reach here
    return "MoveAhead"


# ■■ Main Navigation Loop ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

def run_agent(controller, encoder, predictor, scene_clf, detector,
              narrator, cog_map, n_steps=300, surprise_threshold=0.25):
    """
    Main navigation loop — the heart of Vision-to-Voice.

    Runs the blind student navigation agent for n_steps in the AI2-THOR
    simulation, building a cognitive map in real time and narrating
    discoveries via text-to-speech.

    Pipeline per step (in exact order):
        1. PERCEIVE    — capture frame, encode with DINOv2
        2. SURPRISE    — compute JEPA prediction error (Paper 1)
        3. MAP         — update topological cognitive map (Paper 2)
        4. DETECT      — conditional YOLOE + OCR (Novel: JEPA→YOLOE trigger)
        5. FRONTIER    — JEPA-biased frontier selection (Paper 1 + 2)
        6. FILTER      — Paper 3 action-place compatibility check
        7. STUCK CHECK — escape corners via forced rotation
        8. NARRATE     — periodic voice update (every 5 steps)
        9. EXECUTE     — send action to AI2-THOR

    Args:
        controller         : ai2thor.controller.Controller — live sim
        encoder            : DINOEncoder — frozen DINOv2 backbone
        predictor          : JEPAPredictor — trained world model (eval mode)
        scene_clf          : SceneContextMLP — trained Paper 3 filter (eval mode)
        detector           : ConditionalDetector — YOLOE + EasyOCR
        narrator           : Narrator — non-blocking TTS
        cog_map            : CognitivMap — topological cognitive map (starts empty)
        n_steps            : int — number of exploration steps
        surprise_threshold : float — JEPA error above this triggers YOLOE

    Returns:
        (cog_map, error_log) — the built map and per-step surprise scores
        error_log is used by Riya's dashboard_screen3.py for plotting
    """

    # ── State variables ──────────────────────────────────────────
    prev_cls  = None    # previous step's CLS token — None on first step
    prev_act  = None    # previous step's action string
    prev_nid  = None    # previous step's node ID in the cognitive map

    error_log = []      # per-step surprise scores → dashboard plotting

    # Stuck detection: track recent positions to escape corners
    # Paper 2 (Gupta et al. 2017): agents get stuck in corners of
    # grid-world environments — stuck detection is essential
    recent_positions = deque(maxlen=10)

    # Ablation tracking: count Paper 3 overrides for the final report
    paper3_override_count = 0

    # YOLOE trigger tracking: count how often surprise > threshold
    yoloe_trigger_count = 0

    print(f"\n{'='*60}")
    print(f"  Vision-to-Voice Navigation Agent")
    print(f"  Steps: {n_steps} | Surprise threshold: {surprise_threshold}")
    print(f"{'='*60}\n")

    narrator.say("Vision to Voice system online. Beginning exploration.")

    # ── Main loop ────────────────────────────────────────────────
    for step in range(n_steps):

        # ══════════════════════════════════════════════════════════
        # STEP 1: PERCEIVE
        # Capture RGB frame from AI2-THOR, encode with frozen DINOv2.
        # CLS token (384,) = whole-frame semantic embedding
        # Patch tokens (256, 384) = spatial feature grid
        # ══════════════════════════════════════════════════════════
        raw_frame = controller.last_event.frame              # numpy HxWx3
        frame = Image.fromarray(raw_frame)                   # → PIL Image
        cls, ptch = encoder.encode(frame)                    # → (384,), (256,384)

        # Agent pose from AI2-THOR metadata
        pos = controller.last_event.metadata["agent"]["position"]
        rot = controller.last_event.metadata["agent"]["rotation"]["y"]
        pos2 = [pos["x"], pos["z"]]                          # 2D position for map

        # ══════════════════════════════════════════════════════════
        # STEP 2: COMPUTE JEPA SURPRISE
        # Paper 1 (Mirowski et al. 2017) + JEPA (LeCun 2024):
        #   surprise = MSE(predicted_z_t1, actual_z_t1)
        #   High surprise → unexpected transition → interesting frontier
        #   First step has no previous embedding → surprise = 0.0
        # ══════════════════════════════════════════════════════════
        if prev_cls is not None and prev_act is not None:
            surprise = prediction_error(predictor, prev_cls, prev_act, cls)
        else:
            surprise = 0.0  # first step: no prediction possible yet

        error_log.append(surprise)  # → Riya's dashboard_screen3.py

        # ══════════════════════════════════════════════════════════
        # STEP 3: UPDATE COGNITIVE MAP
        # Paper 2 (Gupta et al. 2017): build topological graph in real
        # time. Each node = (position, rotation, DINOv2 features, surprise).
        # Edges connect consecutive positions → traversability graph.
        # CognitivMap handles deduplication internally (0.25m threshold).
        # ══════════════════════════════════════════════════════════
        nid = cog_map.add_node(pos2, rot, cls, ptch, surprise)

        if prev_nid is not None:
            cog_map.add_edge(prev_nid, nid)  # connect sequential steps

        # ══════════════════════════════════════════════════════════
        # STEP 4: CONDITIONAL YOLOE + OCR (Riya's module)
        # Novel contribution: JEPA surprise gates YOLOE activation.
        # Only run expensive object detection when the world model
        # says "something unexpected happened here."
        # This saves ~70% compute vs running YOLOE every frame.
        # ══════════════════════════════════════════════════════════
        if surprise > surprise_threshold:
            yoloe_trigger_count += 1
            objects, ocr_text = detector.run(frame)

            # OCR text → tag the cognitive map node as a named landmark
            if ocr_text:
                cog_map.tag_label(nid, ocr_text)
                narrator.say(f"Sign detected: {ocr_text}")

            # Object detections → narrate top 3 for the blind user
            if objects:
                narrator.say(f"I can see: {', '.join(objects[:3])}")

        # ══════════════════════════════════════════════════════════
        # STEP 5: FRONTIER SELECTION (Syeda's core contribution)
        # Paper 1 + Paper 2 combined:
        #   Frontiers = nodes with < 3 edges (not fully explored)
        #   Selection = argmax(surprise) over frontiers
        #   → Agent preferentially visits high-surprise areas
        #   → This is the KEY INNOVATION: surprise drives exploration
        # ══════════════════════════════════════════════════════════
        frontiers = cog_map.frontier_nodes()

        if not frontiers:
            # Map is fully explored — all nodes have ≥ 3 edges
            print(f"\n[navigator] Step {step}: No frontiers remain. "
                  f"Exploration complete!")
            narrator.say("Exploration complete. All areas have been mapped.")
            break

        # JEPA-biased selection: pick the frontier with highest surprise
        # This makes the agent "curious" — it goes where things were unexpected
        target = max(
            frontiers,
            key=lambda n: cog_map.G.nodes[n].get("surprise", 0.0)
        )

        # ══════════════════════════════════════════════════════════
        # STEP 6: PAPER 3 FILTER (Syeda's core contribution)
        # Epstein et al. 2019: action-place compatibility.
        # The SceneContextMLP has learned which actions are
        # appropriate in which scenes from DINOv2 features.
        # If the proposed action scores below threshold (0.4),
        # Paper 3 overrides with the highest-scoring alternative.
        # ══════════════════════════════════════════════════════════
        raw_action = plan_action(controller, cog_map, target)
        scene_action = map_to_scene_action(raw_action)

        # Ask the Scene MLP if this action is appropriate here
        filtered_scene_action = scene_clf.filter(
            scene_action, cls, threshold=0.4
        )

        # Check if Paper 3 overrode our planned action
        if filtered_scene_action != scene_action:
            # Map the filtered scene action back to an AI2-THOR action
            action = scene_action_to_navigator(
                filtered_scene_action, raw_action
            )
            paper3_override_count += 1
            print(f"  [Paper 3] Step {step}: Override {raw_action} "
                  f"({scene_action}) → {action} ({filtered_scene_action})")
        else:
            # Paper 3 approved — use the original planned action
            action = raw_action

        # ══════════════════════════════════════════════════════════
        # STEP 7: STUCK DETECTION
        # FloorPlan environments have corners and dead ends.
        # If x-position variance < 0.1 across last 10 steps,
        # the agent is stuck → inject a forced rotation to escape.
        # Paper 2 acknowledges this failure mode in grid worlds.
        # ══════════════════════════════════════════════════════════
        recent_positions.append(pos2)

        if len(recent_positions) == 10:
            xs = [p[0] for p in recent_positions]
            x_var = np.var(xs)
            if x_var < 0.1:
                action = "RotateRight"  # override to escape the corner
                print(f"  [Stuck] Step {step}: x-variance={x_var:.4f} < 0.1 "
                      f"→ injecting RotateRight to escape")

        # ══════════════════════════════════════════════════════════
        # STEP 8: NARRATION (every 5 steps)
        # Periodic voice update so the blind user knows what's happening.
        # Includes: step count, action, map size, surprise level.
        # ══════════════════════════════════════════════════════════
        if step % 5 == 0:
            narrator.say(
                f"Step {step}. Moving {action}. "
                f"{len(cog_map.G.nodes)} locations mapped. "
                f"Surprise level: "
                f"{'high' if surprise > surprise_threshold else 'low'}."
            )

        # ══════════════════════════════════════════════════════════
        # STEP 9: EXECUTE
        # Send the chosen action to AI2-THOR simulation.
        # Update state variables for the next iteration.
        # ══════════════════════════════════════════════════════════
        controller.step(action)

        prev_nid = nid
        prev_cls = cls
        prev_act = action

    # ── Post-loop summary ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Navigation Complete — {len(error_log)} steps executed")
    print(f"  Nodes mapped : {cog_map.node_count()}")
    print(f"  Edges built  : {cog_map.edge_count()}")
    print(f"  YOLOE triggers: {yoloe_trigger_count} "
          f"({100*yoloe_trigger_count/max(len(error_log),1):.1f}%)")
    print(f"  Paper 3 overrides: {paper3_override_count}")
    print(f"{'='*60}\n")

    return cog_map, error_log


# ■■ Standalone Test Block ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

if __name__ == "__main__":
    """
    Standalone integration test — runs the full pipeline on FloorPlan1.

    Usage:
        python -m src.navigator

    Loads all modules from config.yaml, runs 200 steps, prints summary.
    Requires AI2-THOR installed and a display (or Xvfb on Linux).
    """

    # ── Load config ──────────────────────────────────────────────
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "config.yaml"
    )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("  Vision-to-Voice Navigator — Integration Test")
    print("=" * 60)
    print(f"  Config loaded from: {config_path}")

    # ── Read config values (no hardcoded paths) ──────────────────
    cls_dim            = cfg["dino"]["cls_dim"]
    jepa_hidden        = cfg["jepa"]["hidden_dim"]
    jepa_model_path    = cfg["jepa"]["model_path"]
    scene_model_path   = cfg["scene_mlp"]["model_path"]
    surprise_threshold = cfg["mapper"]["surprise_threshold"]
    n_steps            = 200  # integration test: 200 steps on FloorPlan1

    # ── Initialize AI2-THOR ──────────────────────────────────────
    print("\n[1/6] Starting AI2-THOR controller...")
    from ai2thor.controller import Controller

    controller = Controller(
        scene="FloorPlan1",
        width=cfg["ai2thor"]["width"],
        height=cfg["ai2thor"]["height"],
        fieldOfView=cfg["ai2thor"]["fov"],
    )
    print(f"       Scene: FloorPlan1 ({cfg['ai2thor']['width']}x"
          f"{cfg['ai2thor']['height']}, FOV={cfg['ai2thor']['fov']})")

    # ── Initialize Track A: DINOv2 Encoder (Rayyan) ──────────────
    print("\n[2/6] Loading DINOv2 encoder (frozen)...")
    encoder = DINOEncoder()

    # ── Initialize Track B: JEPA Predictor (Syeda) ───────────────
    print("\n[3/6] Loading JEPA world model...")
    predictor = load_jepa(
        model_path=jepa_model_path,
        z_dim=cls_dim,
        act_dim=5,
        hidden=jepa_hidden,
    )
    predictor.eval()  # ensure eval mode — no dropout during inference

    # ── Initialize Track B: Scene Context MLP (Syeda) ────────────
    print("\n[4/6] Loading Scene Context MLP (Paper 3 filter)...")
    scene_clf = SceneContextMLP(input_dim=cls_dim)
    scene_clf.load_state_dict(
        torch.load(scene_model_path, map_location="cpu", weights_only=True)
    )
    scene_clf.eval()  # ensure eval mode — no dropout during inference
    print(f"       Loaded from: {scene_model_path}")

    # ── Initialize Track C: Detector + Narrator (Riya) ───────────
    print("\n[5/6] Loading YOLOE + EasyOCR detector...")
    detector = ConditionalDetector(surprise_threshold=surprise_threshold)

    print("\n[6/6] Initializing narrator (TTS)...")
    narrator = make_narrator(enabled=True)

    # ── Initialize Cognitive Map (empty — built during exploration) ─
    cog_map = CognitivMap()

    # ── Run the agent ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting 200-step exploration run...")
    print("=" * 60)

    cog_map, error_log = run_agent(
        controller=controller,
        encoder=encoder,
        predictor=predictor,
        scene_clf=scene_clf,
        detector=detector,
        narrator=narrator,
        cog_map=cog_map,
        n_steps=n_steps,
        surprise_threshold=surprise_threshold,
    )

    # ── Final Summary ────────────────────────────────────────────
    avg_surprise = np.mean(error_log) if error_log else 0.0
    max_surprise = np.max(error_log) if error_log else 0.0
    labeled = cog_map.labeled_nodes()
    yoloe_triggers = sum(1 for s in error_log if s > surprise_threshold)

    print("\n" + "=" * 60)
    print("  FINAL INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"  Total nodes mapped    : {cog_map.node_count()}")
    print(f"  Total edges           : {cog_map.edge_count()}")
    print(f"  Average surprise      : {avg_surprise:.4f}")
    print(f"  Max surprise          : {max_surprise:.4f}")
    print(f"  Named nodes (OCR)     : {len(labeled)}")
    if labeled:
        for nid, label in labeled:
            print(f"    Node {nid}: \"{label}\"")
    print(f"  YOLOE trigger rate    : {yoloe_triggers}/{len(error_log)} "
          f"({100*yoloe_triggers/max(len(error_log),1):.1f}%)")
    print(f"  Steps executed        : {len(error_log)}")
    print("=" * 60)

    # ── Health checks ────────────────────────────────────────────
    if cog_map.node_count() > 50:
        print("  [OK] Node count > 50 — healthy exploration")
    else:
        print("  [!]  Node count <= 50 — agent may not be exploring enough")

    if 0.10 <= (yoloe_triggers / max(len(error_log), 1)) <= 0.30:
        print("  [OK] YOLOE trigger rate in expected 10-30% range")
    else:
        print("  [!]  YOLOE trigger rate outside expected 10-30% range")

    # ── Cleanup ──────────────────────────────────────────────────
    narrator.say("Navigation test complete. Shutting down.")
    if hasattr(narrator, 'shutdown'):
        narrator.shutdown()
    controller.stop()
    print("\n  Controller stopped. Test complete.\n")
