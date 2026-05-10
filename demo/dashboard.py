# ============================================================
# dashboard.py  |  Demo  |  Vision-to-Voice Live Dashboard
# ============================================================
#
# Three-screen real-time dashboard displayed alongside the
# navigation loop.  Updated every 5 steps via update().
#
# Layout (20×9 figure, dark background #0F172A):
#   Screen 1 (left)   — Agent View + DINOv3 Attention
#   Screen 2 (middle) — Cognitive Map (live graph)
#   Screen 3 (right)  — SmoothGrad (top), JEPA Error +
#                        Action Scores (bottom, side by side)
# ============================================================

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import numpy as np

# ── Colour palette ───────────────────────────────────────────
BG_DARK   = "#0F172A"
BG_PANEL  = "#1E293B"
SPINE_CLR = "#334155"
EDGE_CLR  = "#475569"
TXT_WHITE = "white"

# ── Bar colours for Paper 3 action scores ────────────────────
BAR_COLORS = ["#60A5FA", "#34D399", "#FBBF24"]
ACTION_LABELS = ["MoveAhead", "Stop", "Navigate"]


# ■■ Initialisation ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

_fig = None
_ax_cam = None
_ax_map = None
_ax_sal = None
_ax_err = None
_ax_scores = None


def _style_ax(ax, title="", fontsize=10):
    """Apply dark theme to an axis."""
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=TXT_WHITE, fontsize=fontsize,
                 fontweight="bold", pad=8)
    ax.tick_params(colors=TXT_WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE_CLR)


def init_dashboard():
    """
    Create the 3-screen figure. Call ONCE before the navigation loop.

    Returns the figure so callers can save / show it.
    """
    global _fig, _ax_cam, _ax_map, _ax_sal, _ax_err, _ax_scores

    _fig = plt.figure(figsize=(20, 9), facecolor=BG_DARK)

    gs = GridSpec(2, 3, figure=_fig, hspace=0.35, wspace=0.25)

    # Screen 1 — Agent View (full left column)
    _ax_cam = _fig.add_subplot(gs[:, 0])
    _style_ax(_ax_cam, "Agent View + DINOv3 Attention", fontsize=11)
    _ax_cam.axis("off")

    # Screen 2 — Cognitive Map (full middle column)
    _ax_map = _fig.add_subplot(gs[:, 1])
    _style_ax(_ax_map, "Cognitive Map — Live", fontsize=11)

    # Screen 3 top — SmoothGrad Saliency
    _ax_sal = _fig.add_subplot(gs[0, 2])
    _style_ax(_ax_sal, "SmoothGrad Saliency", fontsize=10)
    _ax_sal.axis("off")

    # Screen 3 bottom — two side-by-side sub-panels
    # Use manual positioning for the two bottom-right panels
    # gs[1, 2] area ≈ x 0.68–1.0, y 0.05–0.45
    _ax_err    = _fig.add_axes([0.68, 0.08, 0.13, 0.33])
    _ax_scores = _fig.add_axes([0.84, 0.08, 0.13, 0.33])
    _style_ax(_ax_err,    "JEPA Prediction Error", fontsize=10)
    _style_ax(_ax_scores, "Action Appropriateness", fontsize=10)

    _fig.suptitle("Vision-to-Voice  ·  Live Dashboard",
                  color=TXT_WHITE, fontsize=16, fontweight="bold",
                  y=0.98)

    plt.ion()
    _fig.canvas.draw()
    _fig.canvas.flush_events()
    return _fig


# ■■ Per-step update ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

def update(frame, attn_map, smap, graph,
           error_history, action_scores,
           current_node_id, ocr_text="",
           yoloe_boxes=None, surprise_threshold=0.25):
    """
    Refresh all 5 axes.  Called every 5 steps inside run_agent().

    Args:
        frame            : PIL Image or numpy (H, W, 3)
        attn_map         : numpy (16, 16) — DINOv3 attention
        smap             : numpy (224, 224) — SmoothGrad saliency
        graph            : nx.Graph — cog_map.G
        error_history    : list[float] — all surprise values so far
        action_scores    : list[float] length 3 — [move_fast, stop_wait, navigate]
        current_node_id  : int — current agent node
        ocr_text         : str — OCR text detected this step (if any)
        yoloe_boxes      : list of (label, (x1, y1, x2, y2)) — optional
        surprise_threshold : float — the calibrated JEPA threshold
    """
    global _ax_cam, _ax_map, _ax_sal, _ax_err, _ax_scores

    frame_np = np.asarray(frame)

    # ── SCREEN 1: Agent View + DINOv3 Attention ──────────────
    _ax_cam.clear()
    _ax_cam.imshow(frame_np)

    if attn_map is not None:
        # Upsample 16×16 → frame size for overlay
        from PIL import Image as _PILImage
        import cv2
        h, w = frame_np.shape[:2]
        attn_resized = cv2.resize(attn_map, (w, h),
                                  interpolation=cv2.INTER_LINEAR)
        _ax_cam.imshow(attn_resized, alpha=0.35, cmap="jet")

    # YOLOE bounding boxes
    if yoloe_boxes:
        for label, (x1, y1, x2, y2) in yoloe_boxes:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="#34D399",
                                 facecolor="none")
            _ax_cam.add_patch(rect)
            _ax_cam.text(x1, y1 - 4, label,
                         color="#34D399", fontsize=8,
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15",
                                   facecolor=BG_PANEL, alpha=0.8))

    # OCR text overlay
    if ocr_text:
        _ax_cam.text(0.5, 0.97, f'OCR: "{ocr_text}"',
                     transform=_ax_cam.transAxes,
                     ha="center", va="top",
                     color="#FBBF24", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor=BG_PANEL, alpha=0.85))

    _ax_cam.set_title("Agent View + DINOv3 Attention",
                      color=TXT_WHITE, fontsize=11, fontweight="bold")
    _ax_cam.axis("off")
    _ax_cam.set_facecolor(BG_PANEL)

    # ── SCREEN 2: Cognitive Map ──────────────────────────────
    _ax_map.clear()

    if graph.number_of_nodes() > 0:
        # Build position dict and surprise colour list
        pos  = {}
        surp = []
        labels_to_draw = {}
        for n in graph.nodes:
            node_data = graph.nodes[n]
            p = node_data.get("pos", np.array([0, 0]))
            pos[n] = (float(p[0]), float(p[1]))
            surp.append(float(node_data.get("surprise", 0.0)))
            lbl = node_data.get("label", "")
            if lbl:
                labels_to_draw[n] = lbl

        # Draw all nodes
        nx.draw_networkx_edges(graph, pos=pos, ax=_ax_map,
                               edge_color=EDGE_CLR, width=1.2,
                               alpha=0.6)
        nx.draw_networkx_nodes(graph, pos=pos, ax=_ax_map,
                               node_color=surp,
                               cmap=plt.cm.RdYlBu_r,
                               node_size=50, edgecolors=SPINE_CLR,
                               linewidths=0.5)

        # Current position — yellow star
        if current_node_id is not None and current_node_id in pos:
            _ax_map.scatter(*pos[current_node_id], s=200,
                            marker="*", color="#FBBF24",
                            edgecolors="white", linewidths=0.8,
                            zorder=5)

        # OCR-labelled nodes — show text
        if labels_to_draw:
            nx.draw_networkx_labels(graph, pos=pos, ax=_ax_map,
                                    labels=labels_to_draw,
                                    font_size=7, font_color="#34D399",
                                    font_weight="bold")

    _ax_map.set_title("Cognitive Map — Live",
                      color=TXT_WHITE, fontsize=11, fontweight="bold")
    _ax_map.set_facecolor(BG_PANEL)
    _ax_map.tick_params(colors=TXT_WHITE, labelsize=7)
    for spine in _ax_map.spines.values():
        spine.set_color(SPINE_CLR)

    # ── SCREEN 3 TOP: SmoothGrad Saliency ────────────────────
    _ax_sal.clear()
    # Resize frame to 224×224 to match smap dimensions exactly
    from PIL import Image as _PILImage
    if hasattr(frame, 'resize'):
        frame_224 = np.array(frame.resize((224, 224)))
    else:
        import cv2
        frame_224 = cv2.resize(frame_np, (224, 224))
    _ax_sal.imshow(frame_224)
    if smap is not None:
        _ax_sal.imshow(smap, alpha=0.55, cmap="jet")
    _ax_sal.set_title("SmoothGrad Saliency",
                      color=TXT_WHITE, fontsize=10, fontweight="bold")
    _ax_sal.axis("off")
    _ax_sal.set_facecolor(BG_PANEL)

    # ── SCREEN 3 BOTTOM LEFT: JEPA Prediction Error ──────────
    _ax_err.clear()
    recent = error_history[-30:] if error_history else []
    if recent:
        x_vals = list(range(len(recent)))
        _ax_err.plot(x_vals, recent,
                     color="#60A5FA", linewidth=1.5, alpha=0.9)

        # Threshold line
        _ax_err.axhline(surprise_threshold, color="#EF4444",
                        linestyle="--", alpha=0.7,
                        label="YOLOE threshold")

        # Fill above threshold
        _ax_err.fill_between(
            x_vals, recent, surprise_threshold,
            where=[e > surprise_threshold for e in recent],
            color="#EF4444", alpha=0.2
        )

        _ax_err.legend(fontsize=7, loc="upper left",
                       facecolor=BG_PANEL, edgecolor=SPINE_CLR,
                       labelcolor=TXT_WHITE)

    _style_ax(_ax_err, "JEPA Prediction Error", fontsize=10)

    # ── SCREEN 3 BOTTOM RIGHT: Paper 3 Action Scores ────────
    _ax_scores.clear()

    if action_scores is not None and len(action_scores) == 3:
        bars = _ax_scores.bar(ACTION_LABELS, action_scores,
                              color=BAR_COLORS, width=0.6,
                              edgecolor=SPINE_CLR, linewidth=0.5)
        # Value labels on top of bars
        for bar, score in zip(bars, action_scores):
            _ax_scores.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{score:.2f}",
                ha="center", va="bottom",
                color=TXT_WHITE, fontsize=9, fontweight="bold"
            )

    _ax_scores.set_ylim(0, 1.15)
    _style_ax(_ax_scores, "Action Appropriateness", fontsize=10)
    _ax_scores.tick_params(axis="x", labelrotation=15)

    # ── Flush ────────────────────────────────────────────────
    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()


# ■■ Standalone test ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

if __name__ == "__main__":
    import os
    import sys
    import random
    import torchvision.transforms as T
    from PIL import Image

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    print("=" * 60)
    print("  Vision-to-Voice Dashboard — Standalone Test")
    print("=" * 60)

    np.random.seed(42)

    # ── Try loading a real campus photo for saliency ─────────
    _photos = []
    for root, dirs, files in os.walk("data/campus_photos"):
        for f in files:
            if f.endswith((".jpg", ".png", ".jpeg")):
                _photos.append(os.path.join(root, f))

    _tf = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
    ])

    if _photos:
        print(f"  Found {len(_photos)} campus photos")
        _real = Image.open(random.choice(_photos)).convert("RGB")
        test_frame = _real
    else:
        print("  No campus photos — using synthetic frame")
        test_frame = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

    # Build frame_tensor from the real PIL frame
    frame_tensor = _tf(test_frame).unsqueeze(0)
    print(f"  frame_tensor mean: {frame_tensor.mean().item():.3f}"
          f"  (should be 0.2–0.7)")

    # Compute real saliency if possible
    try:
        from src.perception import DINOEncoder
        from src.saliency import SaliencyEngine

        print("  Loading DINOv2 + SaliencyEngine ...")
        _enc = DINOEncoder()
        _engine = SaliencyEngine(_enc.model)
        test_smap = _engine.smoothgrad_map(frame_tensor, n_samples=10)
        test_attn = _enc.attention_map(test_frame)
        print(f"  smap range: [{test_smap.min():.3f}, {test_smap.max():.3f}]")
    except Exception as e:
        print(f"  Could not load models: {e} — using fake smap")
        test_smap = np.random.rand(224, 224).astype(np.float32)
        test_attn = np.random.rand(16, 16).astype(np.float32)

    # Build a small graph resembling a corridor walk
    fake_graph = nx.path_graph(8)
    for n in fake_graph.nodes:
        fake_graph.nodes[n]["pos"]      = np.array([n * 0.25, n * 0.15])
        fake_graph.nodes[n]["surprise"] = np.random.rand()
        fake_graph.nodes[n]["label"]    = ""
    fake_graph.nodes[3]["label"] = "Room 204"
    fake_graph.nodes[6]["label"] = "Library"

    fake_errors = list(np.random.rand(30) * 0.5)
    fake_scores = [0.72, 0.28, 0.55]

    # ── Init + render ────────────────────────────────────────
    fig = init_dashboard()

    update(
        frame=test_frame,
        attn_map=test_attn,
        smap=test_smap,
        graph=fake_graph,
        error_history=fake_errors,
        action_scores=fake_scores,
        current_node_id=4,
        ocr_text="Room 204",
        surprise_threshold=0.25,
    )

    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "dashboard_test.png")
    out_path = os.path.abspath(out_path)
    plt.savefig(out_path, dpi=120, facecolor=BG_DARK,
                bbox_inches="tight")
    print(f"\n[OK] Dashboard test saved -> {out_path}")
    print("     Close the window to exit.")

    plt.ioff()
    plt.show()  # blocks until user closes the window
    print("\n  Dashboard test complete.")

