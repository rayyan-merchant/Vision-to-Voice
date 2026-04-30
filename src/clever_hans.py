"""
clever_hans.py — Track C (Riya Bhart)
Vision-to-Voice | FAST NUCES | 6th Semester AI Project

Clever Hans Audit via AttnLRP Attribution Clustering.

HOW IT FITS IN THE PIPELINE:
  - Collects 100+ attribution maps from real navigation decisions (Weeks 3-4).
  - Clusters them with KMeans to find systematic patterns.
  - Produces a written audit report: are clusters attending to real structure
    (door frames, corridors) or spurious shortcuts (carpet colour, wall texture)?
  - This implements Paper 4 (Lapuschkin et al. 2019 — Unmasking Clever Hans)
    using AttnLRP instead of vanilla SpRAy — the methodologically correct
    choice for a ViT backbone like DINOv3.

WHY THIS MATTERS:
  A system that gets high coverage by following carpet colour instead of
  real corridor structure is BRITTLE. The audit catches this before deployment.
  Finding a shortcut is a RESEARCH RESULT, not a failure.

WORKFLOW:
  1. During navigation runs (Week 3 onwards), navigator.py calls:
       collector.record(pil_frame, action, attnlrp_map)
  2. After 100+ decisions, run:
       python clever_hans.py --maps_dir ../outputs/saliency_maps/ --report_dir ../outputs/clusters/
  3. Read the cluster report and cluster visualisations.

PAPER CONNECTION:
  Paper 4 (Lapuschkin et al. 2019) + AttnLRP (ICML 2024, same author).
  AttnLRP is specifically designed for transformer attention layers — vanilla
  SpRAy / LRP fail for ViT attention mechanisms. We use the upgrade.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Data collection helper (used during navigation in navigator.py)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NavigationDecision:
    """One saved navigation decision for the Clever Hans audit."""
    frame_path: str       # path to saved original frame
    action: str           # action chosen ("MoveAhead", "RotateLeft", etc.)
    attnlrp_map_path: str # path to saved .npy attribution map
    step: int = 0
    surprise_score: float = 0.0


class AttributionCollector:
    """
    Collect attribution maps during navigation runs (used by navigator.py).

    Usage in navigator.py:
        from clever_hans import AttributionCollector
        collector = AttributionCollector(save_dir="outputs/saliency_maps")

        # Inside the navigation loop, after saliency_engine.get_map():
        collector.record(
            pil_frame=frame,
            action=chosen_action,
            attnlrp_map=heatmap_array,
            step=step,
            surprise_score=surprise
        )

        # Save the manifest when the run ends:
        collector.save_manifest()
    """

    def __init__(self, save_dir: str = "outputs/saliency_maps"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "frames").mkdir(exist_ok=True)
        self.decisions: list[NavigationDecision] = []
        print(f"[collector] Saving attribution data to: {self.save_dir}")

    def record(
        self,
        pil_frame,           # PIL Image
        action: str,
        attnlrp_map: np.ndarray,
        step: int = 0,
        surprise_score: float = 0.0,
    ):
        """Save a frame + action + attribution map from one navigation step."""
        idx = len(self.decisions)
        frame_path = str(self.save_dir / "frames" / f"frame_{idx:04d}.png")
        map_path = str(self.save_dir / f"attnlrp_{idx:04d}.npy")

        pil_frame.save(frame_path)
        np.save(map_path, attnlrp_map)

        self.decisions.append(NavigationDecision(
            frame_path=frame_path,
            action=action,
            attnlrp_map_path=map_path,
            step=step,
            surprise_score=surprise_score,
        ))

        if (idx + 1) % 20 == 0:
            print(f"[collector] {idx + 1} decisions recorded.")

    def save_manifest(self):
        """Save a JSON manifest of all decisions (for audit reproducibility)."""
        manifest_path = self.save_dir / "manifest.json"
        data = [
            {
                "idx": i,
                "frame_path": d.frame_path,
                "action": d.action,
                "attnlrp_map_path": d.attnlrp_map_path,
                "step": d.step,
                "surprise_score": d.surprise_score,
            }
            for i, d in enumerate(self.decisions)
        ]
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[collector] Manifest saved: {manifest_path} ({len(data)} decisions)")
        return str(manifest_path)

    def __len__(self):
        return len(self.decisions)


# ──────────────────────────────────────────────────────────────────────────────
# Clever Hans Auditor — main analysis class
# ──────────────────────────────────────────────────────────────────────────────

class CleverHansAuditor:
    """
    Cluster AttnLRP attribution maps to detect systematic shortcut learning.

    Step 1: Load saved attribution maps (from AttributionCollector).
    Step 2: Flatten + L2 normalize.
    Step 3: KMeans with silhouette scoring (k = 3, 4, 5 → pick best).
    Step 4: Visualise each cluster (frames + heatmaps side by side).
    Step 5: Write a human-readable audit report.

    Args:
        maps_dir (str): Directory containing .npy AttnLRP maps and manifest.json.
        report_dir (str): Where to write cluster visualisations and report.
        k_range (tuple): Range of k values to try for KMeans.
        min_samples (int): Minimum maps required to run a meaningful audit.
    """

    def __init__(
        self,
        maps_dir: str,
        report_dir: str,
        k_range: tuple = (3, 4, 5),
        min_samples: int = 30,
    ):
        self.maps_dir = Path(maps_dir)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.k_range = k_range
        self.min_samples = min_samples

        # Populated by load()
        self.frames: list = []           # PIL Images
        self.actions: list[str] = []
        self.maps: list[np.ndarray] = []
        self.metadata: list[dict] = []

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_from_manifest(self, manifest_path: Optional[str] = None) -> int:
        """
        Load frames, actions, and attribution maps from a manifest.json.

        Returns the number of samples loaded.
        """
        from PIL import Image

        if manifest_path is None:
            manifest_path = str(self.maps_dir / "manifest.json")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        for entry in manifest:
            try:
                frame = Image.open(entry["frame_path"]).convert("RGB")
                attn_map = np.load(entry["attnlrp_map_path"])
                self.frames.append(frame)
                self.actions.append(entry["action"])
                self.maps.append(attn_map)
                self.metadata.append(entry)
            except Exception as e:
                print(f"[audit] Skipping entry {entry.get('idx', '?')}: {e}")

        print(f"[audit] Loaded {len(self.maps)} samples from {manifest_path}")
        return len(self.maps)

    def load_from_dir(self) -> int:
        """
        Fallback: load all .npy files in maps_dir (no manifest needed).
        Frames and actions will be unavailable for visualization in this mode.
        """
        npy_files = sorted(self.maps_dir.glob("attnlrp_*.npy"))
        for f in npy_files:
            try:
                self.maps.append(np.load(str(f)))
                self.actions.append("unknown")
                self.frames.append(None)
                self.metadata.append({"idx": len(self.maps) - 1})
            except Exception as e:
                print(f"[audit] Could not load {f}: {e}")
        print(f"[audit] Loaded {len(self.maps)} maps from directory scan.")
        return len(self.maps)

    # ── Clustering ────────────────────────────────────────────────────────────

    def run_audit(self) -> dict:
        """
        Full audit pipeline:
          1. Prepare feature matrix (flatten + L2 normalise)
          2. KMeans for each k in k_range, pick best silhouette
          3. Visualise clusters
          4. Generate report

        Returns:
            dict with keys: best_k, silhouette_score, labels, report_path
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        from sklearn.metrics import silhouette_score

        n = len(self.maps)
        if n < self.min_samples:
            print(f"[audit] Only {n} samples — need at least {self.min_samples}. "
                  f"Continue collecting navigation data.")
            return {"error": "insufficient_samples", "n_samples": n}

        print(f"\n[audit] ── Clever Hans Audit ──────────────────────────────")
        print(f"[audit] Samples: {n} | k range: {self.k_range}")

        # Step 1: Feature matrix
        X = np.stack([m.flatten() for m in self.maps])  # (N, 224*224)
        X = normalize(X)  # L2 normalise — CRITICAL, do not skip
        print(f"[audit] Feature matrix: {X.shape} (L2 normalised)")

        # Step 2: KMeans with silhouette scoring
        best_k = self.k_range[0]
        best_score = -1.0
        best_labels = None

        print(f"\n{'k':>4} {'silhouette':>12} {'note':>20}")
        print("─" * 40)
        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            note = "← best" if score > best_score else ""
            print(f"{k:>4} {score:>12.4f} {note:>20}")
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        print(f"\n[audit] Best: k={best_k}, silhouette={best_score:.4f}")
        if best_score > 0.30:
            print("[audit] ✓ Silhouette > 0.30 — clusters are meaningful.")
        elif best_score > 0.15:
            print("[audit] ⚠ Silhouette 0.15-0.30 — weak structure, collect more data.")
        else:
            print("[audit] ✗ Silhouette < 0.15 — clusters overlap. Need more/varied data.")

        # Step 3: Visualise
        self._visualise_clusters(best_labels, best_k)

        # Step 4: Report
        report_path = self._write_report(best_k, best_score, best_labels)

        return {
            "best_k": best_k,
            "silhouette_score": best_score,
            "labels": best_labels.tolist(),
            "n_samples": n,
            "report_path": report_path,
        }

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _visualise_clusters(self, labels: np.ndarray, k: int, n_examples: int = 6):
        """
        For each cluster: show n_examples side-by-side (frame + AttnLRP map).
        Saves one PNG per cluster.
        """
        print(f"\n[audit] Visualising {k} clusters ...")
        for cluster_id in range(k):
            idxs = [i for i, l in enumerate(labels) if l == cluster_id]
            sample_idxs = idxs[:n_examples]
            n_show = len(sample_idxs)

            if n_show == 0:
                continue

            rows = 2  # row 0: frames, row 1: heatmaps
            fig, axes = plt.subplots(rows, n_show, figsize=(3 * n_show, 6))
            if n_show == 1:
                axes = np.expand_dims(axes, 1)

            fig.suptitle(
                f"Cluster {cluster_id}  ({len(idxs)} decisions)\n"
                f"[Inspect: Does attention focus on STRUCTURE or TEXTURE?]",
                fontsize=12, y=1.02
            )

            for j, idx in enumerate(sample_idxs):
                # Row 0: original frame
                ax_frame = axes[0, j]
                if self.frames[idx] is not None:
                    ax_frame.imshow(self.frames[idx])
                else:
                    ax_frame.text(0.5, 0.5, "No frame", ha="center", va="center")
                ax_title = self.actions[idx] if self.actions[idx] != "unknown" else f"step {self.metadata[idx].get('step', idx)}"
                ax_frame.set_title(ax_title, fontsize=9)
                ax_frame.axis("off")

                # Row 1: AttnLRP heatmap
                ax_heat = axes[1, j]
                ax_heat.imshow(self.maps[idx], cmap="jet", vmin=0, vmax=1)
                ax_heat.axis("off")
                if j == 0:
                    ax_heat.set_ylabel("AttnLRP", fontsize=9)

            plt.tight_layout()
            out_path = self.report_dir / f"cluster_{cluster_id}_attnlrp.png"
            plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[audit] Saved cluster {cluster_id} → {out_path}")

    # ── Report ────────────────────────────────────────────────────────────────

    def _write_report(
        self,
        best_k: int,
        best_score: float,
        labels: np.ndarray,
    ) -> str:
        """Write a plain-text Clever Hans audit report."""
        report_path = self.report_dir / "clever_hans_report.txt"

        action_counts = {}
        for i, label in enumerate(labels):
            cluster_actions = action_counts.setdefault(label, {})
            action = self.actions[i] if i < len(self.actions) else "unknown"
            cluster_actions[action] = cluster_actions.get(action, 0) + 1

        lines = [
            "=" * 70,
            "CLEVER HANS AUDIT REPORT",
            "Vision-to-Voice | FAST NUCES",
            "Track C — Riya Bhart",
            "=" * 70,
            "",
            "METHOD: AttnLRP (ICML 2024) — faithful ViT attribution",
            "         Co-authored by Lapuschkin (Paper 4 author)",
            "         Replaces vanilla SpRAy for transformer models.",
            "",
            f"SAMPLES AUDITED : {len(labels)}",
            f"BEST K (clusters): {best_k}",
            f"SILHOUETTE SCORE : {best_score:.4f}",
            "",
            "SILHOUETTE INTERPRETATION:",
            "  > 0.30  → Clusters are meaningful and distinct.",
            "  0.15-0.30 → Weak structure; collect more diverse data.",
            "  < 0.15  → Clusters overlap; insufficient data.",
            "",
            "─" * 70,
            "CLUSTER BREAKDOWN:",
            "",
        ]

        for cluster_id in range(best_k):
            idxs = [i for i, l in enumerate(labels) if l == cluster_id]
            actions = action_counts.get(cluster_id, {})
            dominant_action = max(actions, key=actions.get) if actions else "N/A"
            lines += [
                f"  CLUSTER {cluster_id} ({len(idxs)} decisions)",
                f"    Dominant action : {dominant_action}",
                f"    Action breakdown: {dict(sorted(actions.items(), key=lambda x: -x[1]))}",
                f"    Visual file     : cluster_{cluster_id}_attnlrp.png",
                "",
                f"    ⬜ WHAT TO INSPECT in the heatmap:",
                f"       ✓ LEGITIMATE  — Agent attends to doors, walls, corridor edges,",
                f"                      floor boundaries, navigation-relevant structure.",
                f"       ✗ SHORTCUT    — Agent attends to carpet colour, wall paint,",
                f"                      uniform texture, or fixed decorations.",
                "",
                f"    📝 FINDING: [FILL IN after visual inspection]",
                f"       [e.g. 'Cluster {cluster_id} attends to doorframe edges — LEGITIMATE']",
                f"       [e.g. 'Cluster {cluster_id} attends to carpet texture — CLEVER HANS']",
                "",
                f"    🔧 PROPOSED FIX (if shortcut found):",
                f"       Apply texture randomisation during data collection.",
                f"       Vary floor/wall textures in AI2-THOR between scenes.",
                "",
                "  " + "─" * 66,
                "",
            ]

        lines += [
            "=" * 70,
            "NOTE: A Clever Hans FINDING is not a failure — it is a research result.",
            "Documenting a shortcut demonstrates the system was held to a higher",
            "standard of verification than most navigation systems.",
            "=" * 70,
        ]

        report_text = "\n".join(lines)
        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"\n[audit] Report written → {report_path}")
        print("\n" + report_text)
        return str(report_path)

    # ── Silhouette Plot ───────────────────────────────────────────────────────

    def plot_silhouette_comparison(self, k_scores: dict[int, float]):
        """
        Plot silhouette scores vs k for inclusion in the final report.
        k_scores: {k: silhouette_score}
        """
        ks = list(k_scores.keys())
        scores = list(k_scores.values())

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(ks, scores, color=["#60A5FA" if s == max(scores) else "#94A3B8" for s in scores])
        ax.axhline(0.30, color="#EF4444", linestyle="--", label="Meaningful threshold (0.30)")
        ax.axhline(0.15, color="#F59E0B", linestyle="--", label="Weak threshold (0.15)")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("KMeans Clustering: Silhouette Score vs k\n(AttnLRP Attribution Maps)")
        ax.set_xticks(ks)
        ax.legend()
        plt.tight_layout()
        out = self.report_dir / "silhouette_comparison.png"
        plt.savefig(str(out), dpi=120)
        plt.close()
        print(f"[audit] Silhouette plot → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Clever Hans Audit on AttnLRP maps")
    parser.add_argument(
        "--maps_dir",
        type=str,
        default="../outputs/saliency_maps",
        help="Directory containing .npy AttnLRP maps and manifest.json",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="../outputs/clusters",
        help="Where to write cluster visualisations and report",
    )
    parser.add_argument(
        "--k_min", type=int, default=3, help="Minimum k for KMeans"
    )
    parser.add_argument(
        "--k_max", type=int, default=5, help="Maximum k for KMeans"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.json (auto-detected if not given)",
    )
    args = parser.parse_args()

    auditor = CleverHansAuditor(
        maps_dir=args.maps_dir,
        report_dir=args.report_dir,
        k_range=tuple(range(args.k_min, args.k_max + 1)),
    )

    # Load data
    manifest_path = args.manifest or str(Path(args.maps_dir) / "manifest.json")
    if Path(manifest_path).exists():
        auditor.load_from_manifest(manifest_path)
    else:
        print(f"[audit] manifest.json not found at {manifest_path} — scanning directory.")
        auditor.load_from_dir()

    # Run audit
    results = auditor.run_audit()
    print(f"\n[audit] Results: {results}")
