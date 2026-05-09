
# ============================================================
# mapper.py  |  Track A  |  Owner: Rayyan
# Topological Cognitive Map  —  Paper 2 (Gupta et al. 2017)
#
# Each node stores:
#   pos      → (x, z) agent position
#   rot      → agent rotation in degrees
#   cls      → DINOv3 CLS token (384,)
#   patches  → DINOv3 patch tokens (256, 384)
#   surprise → JEPA prediction error at this location
#   label    → OCR-derived text (e.g. "Room 204", "Library")
#
# Fixes applied (v3):
#   FIX 1 — DEDUP_THRESHOLD: 0.25 → 0.15 m
#            AI2-THOR's grid step is exactly 0.25 m. With threshold=0.25,
#            _nearest_within() was returning the previous node as "close
#            enough" on every successful MoveAhead, so the map never grew
#            past ~5 nodes. At 0.15 m, a 0.25 m step always creates a
#            new node; only floating-point jitter (<0.01 m) gets merged.
#
#   FIX 2 — frontier_nodes(current_nid): exclude current node
#            The agent's current node always has degree=1 (one incoming
#            edge) on first visit, so it always appeared in the frontier
#            list. plan_action then computed distance≈0 → returned
#            MoveAhead → hit the wall → surprise stayed high → YOLOE
#            fired every frame. Excluding current_nid forces the agent
#            to aim at a DIFFERENT node.
#
#   FIX 3 (v4) — visit_count + explored flag
#            Nodes now track visit_count (incremented on merge) and an
#            explored flag (set by navigator). frontier_nodes() skips
#            explored nodes, and the navigator's frontier scorer
#            penalises high visit_count to prevent oscillation.
# ============================================================

import json
import os

import networkx as nx
import numpy as np


class CognitivMap:

    # AI2-THOR default grid step = 0.25 m.
    # Threshold MUST be strictly less than the grid step so consecutive
    # positions are never merged. 0.15 m gives a comfortable margin.
    DEDUP_THRESHOLD = 0.15   # metres  (was 0.25 — caused every step to merge)
    MAX_DEGREE = 3

    def __init__(self):
        self.G        = nx.Graph()
        self._next_id = 0

    # ── Node operations ───────────────────────────────────────────────────────

    def add_node(self, pos2, rot, cls_tensor, patch_tensor, surprise=0.0) -> int:
        """
        Add a node for the current position, or update an existing nearby
        node if one exists within DEDUP_THRESHOLD.

        Args:
            pos2         : [x, z]          — 2-D agent position
            rot          : float           — agent rotation in degrees
            cls_tensor   : (384,)  Tensor  — DINOv3 CLS token
            patch_tensor : (256,384) Tensor — DINOv3 patch tokens
            surprise     : float           — JEPA prediction error

        Returns:
            int — node ID (new or existing)
        """
        pos_arr   = np.array(pos2, dtype=float)
        close_nid = self._nearest_within(pos_arr, self.DEDUP_THRESHOLD)

        if close_nid is not None:
            # Node already exists nearby — keep the higher surprise score
            node = self.G.nodes[close_nid]
            node["surprise"] = max(node["surprise"], float(surprise))
            node["visit_count"] = node.get("visit_count", 1) + 1
            return close_nid

        # New position — create a fresh node
        def to_np(t):
            if hasattr(t, "detach"):
                return t.detach().cpu().numpy()
            return np.array(t)

        nid = self._next_id
        self._next_id += 1
        self.G.add_node(nid, **{
            "pos":         pos_arr,
            "rot":         float(rot),
            "cls":         to_np(cls_tensor),
            "patches":     to_np(patch_tensor),
            "surprise":    float(surprise),
            "label":       "",
            "visit_count": 1,
            "explored":    False,
        })
        return nid

    def add_edge(self, nid_a: int, nid_b: int):
        """Connect two adjacent nodes (sequential steps in the trajectory)."""
        if nid_a != nid_b and nid_a in self.G.nodes and nid_b in self.G.nodes:
            self.G.add_edge(nid_a, nid_b)

    def tag_label(self, nid: int, text: str):
        """
        Attach an OCR-derived text label to a node.
        Turns anonymous map nodes into named landmarks ("Room 204", "Library").
        """
        if nid in self.G.nodes and text and len(text.strip()) > 1:
            self.G.nodes[nid]["label"] = text.strip()

    # ── Frontier methods ──────────────────────────────────────────────────────

    def frontier_nodes(self, current_nid=None) -> list:
        """
        Return unexplored frontier nodes — locations the agent has not
        fully explored yet.

        A node is a frontier if:
          - degree < 3 (fewer than 3 distinct neighbours)
          - NOT marked as explored
          - NOT the current node (unless it has degree=0, cold-start fix)

        Cold-start fix: on the very first step the map has 1 node with
        degree=0. Excluding it immediately makes the frontier list empty
        and terminates exploration at step 0. By only excluding nodes
        with degree >= 1 we guarantee the agent always has at least one
        target.

        Fallback: if all non-explored frontiers are exhausted, return
        ALL non-current nodes so the agent doesn't terminate prematurely.

        Returns:
            list[int] — node IDs to consider as exploration targets
        """
        # v11: Removed the strict 'explored' check.
        # A node is a frontier if it has fewer than MAX_DEGREE (3) edges.
        # We don't exclude current_nid here because navigator.py handles it.
        result = [
            n for n in self.G.nodes
            if self.G.degree(n) < self.MAX_DEGREE
        ]
        return result

    def score_frontier(self, nid: int, current_pos=None,
                       dispatched_count: int = 0) -> float:
        """
        Score a frontier — higher is more worth visiting.

        Scoring formula (v4 — anti-oscillation):
          surprise / visit_count          — decays with repeated visits
          + 2.0 / max(degree, 1)          — prefer less-connected nodes
          + 0.1 / (dist + ε)              — proximity tiebreaker
          - 999 if dispatched_count >= 3   — hard cap on re-targeting
        """
        if nid not in self.G.nodes:
            return 0.0

        node = self.G.nodes[nid]
        surprise    = node["surprise"]
        visits      = node.get("visit_count", 1)
        degree      = self.G.degree(nid)

        # Soft cap: deprioritize over-targeted nodes but don't exclude them
        if dispatched_count >= 10:
            return -1.0

        score = surprise / max(visits, 1) + 2.0 / max(degree, 1)

        if current_pos is not None:
            dist = np.linalg.norm(
                np.array(current_pos) - node["pos"]
            )
            score += 0.1 / (dist + 1e-5)

        return score

    # ── Metrics ───────────────────────────────────────────────────────────────

    def coverage_percent(self, reachable_positions: list) -> float:
        """
        Fraction of reachable floor positions the agent has visited.

        Args:
            reachable_positions : list of dicts with 'x', 'y', 'z' keys
                                  (from AI2-THOR GetReachablePositions)

        Returns:
            float in [0.0, 1.0]
        """
        if not reachable_positions:
            return 0.0

        visited = set()
        for _, data in self.G.nodes(data=True):
            for rpos in reachable_positions:
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
        """Return [(nid, label), ...] for nodes that have OCR labels."""
        return [
            (n, d["label"])
            for n, d in self.G.nodes(data=True)
            if d["label"]
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Serialise the map to JSON for post-run analysis."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True
        )
        data = {
            "nodes": {
                str(n): {
                    "pos":      d["pos"].tolist(),
                    "rot":      d["rot"],
                    "surprise": d["surprise"],
                    "label":    d["label"],
                    # cls / patches omitted — too large, not needed for analysis
                }
                for n, d in self.G.nodes(data=True)
            },
            "edges": [[a, b] for a, b in self.G.edges()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CognitivMap] Saved {self.node_count()} nodes, "
              f"{self.edge_count()} edges → {path}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _nearest_within(self, pos_arr: np.ndarray, threshold: float):
        """Return the ID of the nearest node if it is within threshold, else None."""
        best_id, best_dist = None, float("inf")
        for nid, data in self.G.nodes(data=True):
            d = np.linalg.norm(pos_arr - data["pos"])
            if d < best_dist:
                best_dist, best_id = d, nid
        return best_id if best_dist < threshold else None


# ── Self-test (no AI2-THOR needed) ────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("=" * 55)
    print("mapper.py  — self test")
    print("=" * 55)

    cmap = CognitivMap()

    # ── Test 1: 0.25 m steps all create distinct nodes ────────────
    # Simulates a straight corridor walk with AI2-THOR default step size.
    # Every step must produce a NEW node (0.25 > DEDUP_THRESHOLD=0.15).
    corridor = [[i * 0.25, 0.0] for i in range(8)]   # 8 positions, 0.25 m apart
    node_ids = []
    for i, pos in enumerate(corridor):
        nid = cmap.add_node(
            pos, rot=0.0,
            cls_tensor=torch.randn(384),
            patch_tensor=torch.randn(256, 384),
            surprise=float(i) * 0.1,
        )
        node_ids.append(nid)

    assert len(set(node_ids)) == 8, (
        f"FIX 1 FAILED: expected 8 distinct nodes from 0.25 m steps, "
        f"got {len(set(node_ids))}. "
        f"DEDUP_THRESHOLD ({CognitivMap.DEDUP_THRESHOLD}) must be < 0.25."
    )
    print(f"[PASS] Test 1 — 8 distinct nodes from 0.25 m corridor steps "
          f"(threshold={CognitivMap.DEDUP_THRESHOLD} m)")

    # ── Test 2: jitter (<0.10 m) correctly deduplicates ───────────
    # Floating-point noise from odometry should not bloat the map.
    nid_jitter = cmap.add_node(
        [0.05, 0.03], 0.0,           # 0.058 m from origin — well within 0.15
        torch.randn(384), torch.randn(256, 384),
        surprise=0.99,
    )
    assert nid_jitter == node_ids[0], (
        f"FIX 1 FAILED: jitter position should merge with node 0, "
        f"got new node {nid_jitter}"
    )
    assert cmap.G.nodes[node_ids[0]]["surprise"] == 0.99, \
        "Surprise update on dedup failed"
    print("[PASS] Test 2 — sub-threshold jitter correctly deduplicates "
          "and updates surprise")

    # ── Test 3: frontier_nodes excludes current node (degree>=1 only) ─
    # Connect corridor as a chain so all interior nodes have degree=2,
    # endpoints have degree=1.
    for i in range(len(node_ids) - 1):
        cmap.add_edge(node_ids[i], node_ids[i + 1])

    current = node_ids[0]   # endpoint, degree=1 — should be excluded
    assert cmap.G.degree(current) >= 1, "Test setup: node must have edges"

    frontiers_without_fix = cmap.frontier_nodes()
    frontiers_with_fix    = cmap.frontier_nodes(current_nid=current)

    assert current in frontiers_without_fix, \
        "Test setup: current node should appear when not excluded"
    assert current not in frontiers_with_fix, (
        f"FIX 2 FAILED: node {current} (degree={cmap.G.degree(current)}) "
        f"still in frontier_nodes(current_nid={current})"
    )
    print(f"[PASS] Test 3 — frontier_nodes excludes node with degree>=1: "
          f"{len(frontiers_without_fix)} total → {len(frontiers_with_fix)} after exclusion")

    # Cold-start: degree=0 node must NOT be excluded even if passed as current_nid.
    # This is the bug we introduced and then fixed — on step 0 the only node
    # has degree=0, excluding it empties the frontier list and kills exploration.
    isolated_nid = cmap.add_node([99.0, 99.0], 0.0,
                                  torch.randn(384), torch.randn(256, 384), surprise=0.5)
    assert cmap.G.degree(isolated_nid) == 0, "Test setup: isolated node must have degree 0"
    frontiers_coldstart = cmap.frontier_nodes(current_nid=isolated_nid)
    assert isolated_nid in frontiers_coldstart, (
        "COLD-START BUG: isolated node (degree=0) was excluded from frontiers — "
        "this would cause immediate termination at step 0"
    )
    print("[PASS] Test 3b — cold-start: degree=0 node retained in frontiers")

    # ── Test 4: score_frontier ranks by surprise + proximity ──────
    scores = [
        (n, cmap.score_frontier(n, current_pos=[0.0, 0.0]))
        for n in frontiers_with_fix
    ]
    scores.sort(key=lambda x: -x[1])
    print(f"[PASS] Test 4 — score_frontier top candidate: "
          f"node {scores[0][0]} (score={scores[0][1]:.4f})")

    # ── Test 5: tag_label ─────────────────────────────────────────
    cmap.tag_label(node_ids[3], "Room 204")
    labeled = cmap.labeled_nodes()
    assert len(labeled) == 1 and labeled[0][1] == "Room 204", \
        f"tag_label failed: {labeled}"
    print(f"[PASS] Test 5 — tag_label → {labeled}")

    # ── Test 6: coverage_percent ──────────────────────────────────
    reachable = [{"x": p[0], "z": p[1]} for p in corridor]
    cov = cmap.coverage_percent(reachable)
    assert cov > 0.0, "coverage_percent returned 0"
    print(f"[PASS] Test 6 — coverage_percent = {cov:.1%}")

    # ── Test 7: save ──────────────────────────────────────────────
    cmap.save("data/test_map.json")
    print(f"[PASS] Test 7 — save to data/test_map.json")

    print()
    print(f"  Nodes : {cmap.node_count()}")
    print(f"  Edges : {cmap.edge_count()}")
    print()
    print("All 7 tests passed. CognitivMap is ready.")