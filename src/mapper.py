# ============================================================
# mapper.py  |  Track A  |  Owner: Rayyan
# Topological Cognitive Map  —  implements Paper 2 (Gupta et al. 2017)
# as a topological graph instead of a metric grid.
#
# Each node stores:
#   pos      → (x, z) agent position
#   rot      → agent rotation in degrees
#   cls      → DINOv3 CLS token (384,)
#   patches  → DINOv3 patch tokens (256, 384)
#   surprise → JEPA prediction error at this location
#   label    → OCR-derived text (e.g. "Room 204", "Library")
# ============================================================

import json
import os

import networkx as nx
import numpy as np


class CognitivMap:
    """
    Topological cognitive map — grows in real time from scratch.

    Key design decisions (justify these in presentation):
      - Topological (not metric) → robust to odometry drift
      - DINOv3 features at each node → semantic, not just geometric
      - OCR labels → named landmark graph
      - JEPA surprise scores → high-value frontiers highlighted
    """

    DEDUP_THRESHOLD = 0.25  # metres — AI2-THOR grid step is ~0.25m, so this creates
    #                          a new node every ~1 step rather than merging everything

    def __init__(self):
        self.G        = nx.Graph()
        self._next_id = 0

    # ── Node operations ───────────────────────────────────────────────────────

    def add_node(self, pos2, rot, cls_tensor, patch_tensor, surprise=0.0) -> int:
        """
        Add a node for the current position.

        If a node already exists within DEDUP_THRESHOLD, update its
        surprise (keep max) and return the existing node ID instead of
        creating a duplicate. This prevents the map bloating with tiny
        position differences.

        Args:
            pos2         : [x, z] — agent position
            rot          : float  — agent rotation degrees
            cls_tensor   : (384,) torch.Tensor or numpy array
            patch_tensor : (256, 384) torch.Tensor or numpy array
            surprise     : float — JEPA prediction error (0 on first step)

        Returns:
            int — node ID
        """
        pos_arr = np.array(pos2, dtype=float)

        # Check for nearby existing node
        close_nid = self._nearest_within(pos_arr, self.DEDUP_THRESHOLD)
        if close_nid is not None:
            # Keep the higher surprise value
            existing = self.G.nodes[close_nid]["surprise"]
            self.G.nodes[close_nid]["surprise"] = max(existing, float(surprise))
            return close_nid

        # Convert tensors to numpy for JSON-serialisable storage
        def to_np(t):
            if hasattr(t, "detach"):
                return t.detach().cpu().numpy()
            return np.array(t)

        nid = self._next_id
        self._next_id += 1

        self.G.add_node(nid, **{
            "pos":      pos_arr,
            "rot":      float(rot),
            "cls":      to_np(cls_tensor),     # (384,)
            "patches":  to_np(patch_tensor),   # (256, 384)
            "surprise": float(surprise),
            "label":    "",                    # filled in by EasyOCR
        })
        return nid

    def add_edge(self, nid_a: int, nid_b: int):
        """Connect two adjacent nodes."""
        if nid_a != nid_b and nid_a in self.G.nodes and nid_b in self.G.nodes:
            self.G.add_edge(nid_a, nid_b)

    def tag_label(self, nid: int, text: str):
        """
        Attach an OCR-derived text label to a node.
        This is what makes the cognitive map SEMANTIC:
        nodes become named landmarks like "Room 204" or "Library".
        """
        if nid in self.G.nodes and text and len(text.strip()) > 1:
            self.G.nodes[nid]["label"] = text.strip()

    # ── Frontier methods ──────────────────────────────────────────────────────

    def frontier_nodes(self) -> list:
        """
        Return nodes with degree < 3 (fewer than 3 explored neighbours).
        These are the exploration targets — the agent hasn't fully
        explored the space around them yet.

        If no frontiers exist, the map is fully explored.
        """
        return [n for n in self.G.nodes if self.G.degree(n) < 3]

    def score_frontier(self, nid: int, current_pos=None) -> float:
        """
        Score a frontier node — higher = more worth visiting.

        Components:
          surprise   → JEPA says something unexpected happened here
          proximity  → slight preference for closer nodes (avoids huge detours)

        The surprise component dominates. A high-surprise frontier far away
        is still preferred over a boring nearby one.
        """
        if nid not in self.G.nodes:
            return 0.0

        surprise = self.G.nodes[nid]["surprise"]

        if current_pos is not None:
            dist = np.linalg.norm(
                np.array(current_pos) - self.G.nodes[nid]["pos"]
            )
            return surprise + 0.1 / (dist + 1e-5)

        return surprise

    # ── Metrics ───────────────────────────────────────────────────────────────

    def coverage_percent(self, reachable_positions: list) -> float:
        """
        Fraction of reachable floor positions that the agent has visited.

        Args:
            reachable_positions: list of dicts from GetReachablePositions
                                 each dict has keys 'x', 'y', 'z'

        Returns:
            float in [0, 1]
        """
        if not reachable_positions:
            return 0.0

        visited = set()
        for nid, data in self.G.nodes(data=True):
            for rpos in reachable_positions:
                # GetReachablePositions returns {'x':..,'y':..,'z':..}
                rx = rpos.get("x", rpos[0] if isinstance(rpos, (list, tuple)) else 0)
                rz = rpos.get("z", rpos[1] if isinstance(rpos, (list, tuple)) else 0)
                if np.linalg.norm(data["pos"] - np.array([rx, rz])) < 0.3:
                    visited.add((round(rx, 2), round(rz, 2)))
                    break

        return len(visited) / len(reachable_positions)

    def node_count(self) -> int:
        return self.G.number_of_nodes()

    def edge_count(self) -> int:
        return self.G.number_of_edges()

    def labeled_nodes(self) -> list:
        """Return list of (nid, label) for nodes that have OCR labels."""
        return [
            (n, d["label"])
            for n, d in self.G.nodes(data=True)
            if d["label"]
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save map to JSON (for post-run analysis, not for live use)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "nodes": {
                str(n): {
                    "pos":      d["pos"].tolist(),
                    "rot":      d["rot"],
                    "surprise": d["surprise"],
                    "label":    d["label"],
                    # NOTE: cls and patches are NOT saved (too large, not needed for analysis)
                }
                for n, d in self.G.nodes(data=True)
            },
            "edges": [[a, b] for a, b in self.G.edges()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CognitivMap] Saved {self.node_count()} nodes, {self.edge_count()} edges → {path}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _nearest_within(self, pos_arr: np.ndarray, threshold: float):
        """Return nearest node ID if within threshold, else None."""
        best_id, best_dist = None, float("inf")
        for nid, data in self.G.nodes(data=True):
            d = np.linalg.norm(pos_arr - data["pos"])
            if d < best_dist:
                best_dist, best_id = d, nid
        return best_id if best_dist < threshold else None


# ── Self-test (no AI2-THOR needed) ────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("=" * 50)
    print("mapper.py  — self test (no AI2-THOR needed)")
    print("=" * 50)

    cmap = CognitivMap()

    # Add 10 nodes at different positions
    positions = [
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [3.0, 0.0],
    ]
    node_ids = []
    for i, pos in enumerate(positions):
        cls     = torch.randn(384)
        patches = torch.randn(256, 384)
        nid = cmap.add_node(pos, rot=float(i * 36), cls_tensor=cls,
                            patch_tensor=patches, surprise=float(i) * 0.05)
        node_ids.append(nid)

    # Add edges along the grid
    for i in range(len(node_ids) - 1):
        cmap.add_edge(node_ids[i], node_ids[i + 1])

    # Test deduplication — adding a node 0.1m from node 0 should NOT create new node
    cls2 = torch.randn(384)
    pat2 = torch.randn(256, 384)
    nid_dup = cmap.add_node([0.05, 0.05], 0.0, cls2, pat2, surprise=0.99)
    assert nid_dup == node_ids[0], "Dedup failed — should return existing node"
    assert cmap.G.nodes[node_ids[0]]["surprise"] == 0.99, "Surprise update failed"
    print("[PASS] Deduplication works")

    # Test frontier detection
    frontiers = cmap.frontier_nodes()
    assert len(frontiers) > 0, "No frontiers found"
    print(f"[PASS] frontier_nodes() → {len(frontiers)} frontiers found")

    # Test scoring
    scores = [(n, cmap.score_frontier(n, current_pos=[1.0, 1.0])) for n in frontiers]
    scores.sort(key=lambda x: -x[1])
    print(f"[PASS] score_frontier() → top frontier is node {scores[0][0]} (score={scores[0][1]:.4f})")

    # Test labelling
    cmap.tag_label(node_ids[3], "Room 204")
    labeled = cmap.labeled_nodes()
    assert len(labeled) == 1 and labeled[0][1] == "Room 204"
    print(f"[PASS] tag_label() → labeled node: {labeled}")

    # Test coverage
    reachable = [{"x": p[0], "z": p[1]} for p in positions]
    cov = cmap.coverage_percent(reachable)
    print(f"[PASS] coverage_percent() → {cov:.1%}")

    # Test save
    cmap.save("data/test_map.json")
    print(f"[PASS] save() → data/test_map.json written")

    print()
    print(f"  Total nodes : {cmap.node_count()}")
    print(f"  Total edges : {cmap.edge_count()}")
    print()
    print("All tests passed. CognitivMap is ready.")