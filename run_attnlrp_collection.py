# run_attnlrp_collection.py
# ============================================================
# Automated AttnLRP data collection across multiple scenes.
# Uses MockController when AI2-THOR is unavailable (Windows).
# Target: 100+ diverse samples for Week 5 Clever Hans audit.
# ============================================================

import argparse
import os
import sys
import yaml
import torch
import traceback

from src.mock_controller import MockController
from src.collect_attnlrp import AttnLRPCollector
from src.saliency import SaliencyEngine
from src.perception import DINOEncoder
from src.predictor import load_jepa
from src.scene_classifier import SceneContextMLP
from src.detector import ConditionalDetector
from src.narrator import make_narrator
from src.mapper import CognitivMap
from src.navigator import run_agent, calibrate_surprise_threshold


def run_collection(scenes, steps_per_scene, output_dir):
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("  Vision-to-Voice: AttnLRP Data Collection")
    print(f"  Target scenes   : {scenes}")
    print(f"  Steps per scene : {steps_per_scene}")
    print(f"  Expected total  : {len(scenes) * steps_per_scene} frames")
    print(f"  Output          : {output_dir}")
    print("=" * 60)

    # ── 1. Load all models once ──────────────────────────────────
    print("\n[1/4] Loading models...")
    encoder = DINOEncoder()
    saliency_engine = SaliencyEngine(encoder.model)

    predictor = load_jepa(
        model_path=cfg["jepa"]["model_path"],
        z_dim=cfg["dino"]["cls_dim"],
        act_dim=5,
        hidden=cfg["jepa"]["hidden_dim"],
    )
    predictor.eval()

    scene_clf = SceneContextMLP(input_dim=cfg["dino"]["cls_dim"])
    scene_clf.load_state_dict(
        torch.load(cfg["scene_mlp"]["model_path"], map_location="cpu", weights_only=True)
    )
    scene_clf.eval()

    detector = ConditionalDetector()
    narrator = make_narrator(enabled=False)  # silent for fast collection

    # ── 2. Init collector (auto-resumes if prior data exists) ────
    print("\n[2/4] Initializing collector...")
    collector = AttnLRPCollector(base_dir=output_dir, flush_every=10)

    # ── 3. Iterate scenes ────────────────────────────────────────
    print("\n[3/4] Running navigation sessions...")
    for i, scene in enumerate(scenes, 1):
        print(f"\n{'='*60}")
        print(f"  Scene {i}/{len(scenes)}: {scene}")
        print(f"  Samples so far: {collector.collector_count}")
        print(f"{'='*60}")

        try:
            from ai2thor.controller import Controller
            controller = Controller(
                scene=scene,
                width=cfg["ai2thor"]["width"],
                height=cfg["ai2thor"]["height"],
                fieldOfView=cfg["ai2thor"]["fov"],
            )
            print(f"  [OK] AI2-THOR Controller loaded")
        except Exception as e:
            print(f"  [AI2-THOR ERROR] {type(e).__name__}: {e}")
            print(f"  [INFO] AI2-THOR unavailable. Using MockController.")
            controller = MockController(
                scene=scene,
                width=cfg["ai2thor"]["width"],
                height=cfg["ai2thor"]["height"],
                fieldOfView=cfg["ai2thor"]["fov"],
            )

        try:
            # Calibrate surprise threshold
            threshold = calibrate_surprise_threshold(
                predictor, encoder, controller, n_warmup=30
            )

            cog_map = CognitivMap()

            run_agent(
                controller=controller,
                encoder=encoder,
                predictor=predictor,
                scene_clf=scene_clf,
                detector=detector,
                narrator=narrator,
                cog_map=cog_map,
                n_steps=steps_per_scene,
                surprise_threshold=threshold,
                collector=collector,
                saliency_engine=saliency_engine,
            )
        except Exception as e:
            print(f"\n  [ERROR] Scene {scene} failed: {e}")
            traceback.print_exc()
            # Flush what we have so far — don't lose data
            collector.flush()
            print(f"  [SAVED] Emergency flush — {collector.collector_count} records safe")
        finally:
            controller.stop()

    # ── 4. Final flush and summary ───────────────────────────────
    collector.flush()
    print(f"\n{'='*60}")
    print(f"  Collection Complete!")
    print(f"  Total frames saved : {collector.collector_count}")
    print(f"  Output directory   : {output_dir}")

    # Quick diversity check
    if collector.metadata_list:
        scenes_seen = set(m["scene"] for m in collector.metadata_list)
        actions_seen = set(m["action"] for m in collector.metadata_list)
        surprises = [m["surprise"] for m in collector.metadata_list]
        print(f"  Unique scenes      : {len(scenes_seen)} ({', '.join(scenes_seen)})")
        print(f"  Unique actions     : {len(actions_seen)} ({', '.join(actions_seen)})")
        print(f"  Surprise range     : [{min(surprises):.4f}, {max(surprises):.4f}]")

    status = "PASS" if collector.collector_count >= 100 else "NEED MORE"
    print(f"  100+ target        : [{status}] ({collector.collector_count} collected)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AttnLRP data collection for Clever Hans audit")
    parser.add_argument("--scenes", nargs="+",
                        default=["FloorPlan1", "FloorPlan5", "FloorPlan201", "FloorPlan301"],
                        help="AI2-THOR scene names (or MockController scene labels)")
    parser.add_argument("--steps", type=int, default=35,
                        help="Navigation steps per scene (frames collected = scenes × steps)")
    parser.add_argument("--out", type=str, default="data/saved_attnlrp_maps",
                        help="Output directory")
    args = parser.parse_args()

    run_collection(args.scenes, args.steps, args.out)
