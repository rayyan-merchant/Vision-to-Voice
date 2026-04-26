# ============================================================
# dashboard_screen1.py  |  Track A  |  Owner: Rayyan
#
# Agent View panel — left screen of the 3-screen demo dashboard.
# Shows:
#   • Live RGB frame from AI2-THOR
#   • DINOv3 attention heatmap overlay (semi-transparent)
#   • YOLOE bounding boxes when triggered (green)
#   • OCR text popup when sign detected
#   • Current action + surprise score in title
#
# Used by:  demo/dashboard.py (Week 6 assembly)
# Tested by: run this file directly for a self-test with a dummy frame
# ============================================================


import os
import sys
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class AgentViewPanel:
    """
    Manages the left screen of the Vision-to-Voice dashboard.

    Usage:
        fig, ax = plt.subplots(...)
        panel = AgentViewPanel(ax)
        panel.render(frame, attn_map, action, yolo_boxes, ocr_text, surprise)
    """

    # Colour constants (match the dark dashboard theme)
    BG_COLOR    = "#1E293B"
    SPINE_COLOR = "#334155"
    BOX_COLOR   = "#22C55E"   # green for YOLOE detections
    TEXT_COLOR  = "#F8FAFC"
    HI_SURP     = "#EF4444"   # red title when surprise is high
    LO_SURP     = "#60A5FA"   # blue title when surprise is normal

    SURPRISE_THRESHOLD = 0.25

    def __init__(self, ax):
        self.ax = ax
        self._style_ax()

    def _style_ax(self):
        self.ax.set_facecolor(self.BG_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color(self.SPINE_COLOR)
        self.ax.tick_params(colors=self.TEXT_COLOR)
        self.ax.axis("off")

    def render(
        self,
        frame_pil,          # PIL Image  (224×224)
        attn_map_16x16,     # np.ndarray (16, 16)  — DINOv3 attention
        action_str: str,    # e.g. "MoveAhead"
        yolo_boxes=None,    # list of (label, x1, y1, x2, y2) or None
        ocr_text: str = "", # OCR-detected sign text (empty = no sign)
        surprise: float = 0.0,
    ):
        """
        Render one frame into the panel.

        Call this inside your animation/update loop. Clears and redraws
        the axis on every call so it stays live.
        """
        self.ax.clear()
        self._style_ax()

        # ── Base frame ────────────────────────────────────────────────────────
        frame_arr = np.array(frame_pil.convert("RGB"))
        self.ax.imshow(frame_arr, extent=[0, 224, 224, 0], aspect="auto")

        # ── DINOv3 attention overlay ──────────────────────────────────────────
        attn = self._upscale_attn(attn_map_16x16)
        self.ax.imshow(
            attn, alpha=0.38, cmap="jet",
            vmin=0, vmax=1, extent=[0, 224, 224, 0], aspect="auto"
        )

        # ── YOLOE bounding boxes ──────────────────────────────────────────────
        if yolo_boxes:
            for item in yolo_boxes:
                label, x1, y1, x2, y2 = item
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.8,
                    edgecolor=self.BOX_COLOR,
                    facecolor="none",
                    zorder=5,
                )
                self.ax.add_patch(rect)
                # Label above the box
                self.ax.text(
                    x1 + 2, max(y1 - 3, 4),
                    label,
                    color=self.BOX_COLOR,
                    fontsize=7,
                    fontweight="bold",
                    zorder=6,
                )

        # ── OCR text popup ────────────────────────────────────────────────────
        if ocr_text:
            self.ax.text(
                112, 210,          # bottom-centre of the frame
                f'📋 "{ocr_text}"',
                color="#FCD34D",   # amber
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#1E293B",
                    edgecolor="#FCD34D",
                    alpha=0.85,
                ),
                zorder=7,
            )

        # ── YOLOE trigger indicator ───────────────────────────────────────────
        if surprise > self.SURPRISE_THRESHOLD:
            self.ax.text(
                2, 2,
                "⚡ YOLOE ACTIVE",
                color="#22C55E",
                fontsize=7,
                fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E293B", alpha=0.8),
                zorder=7,
            )

        # ── Title bar ────────────────────────────────────────────────────────
        title_color = self.HI_SURP if surprise > self.SURPRISE_THRESHOLD else self.LO_SURP
        self.ax.set_title(
            f"⬤  {action_str:<13} │  surprise = {surprise:.3f}",
            color=title_color,
            fontsize=8.5,
            pad=5,
            fontweight="bold",
        )

        # ── Axis label ────────────────────────────────────────────────────────
        self.ax.set_xlabel(
            "Agent View  +  DINOv3 Attention",
            color="#64748B",
            fontsize=7,
            labelpad=3,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _upscale_attn(attn_16x16: np.ndarray) -> np.ndarray:
        """Bicubic upscale (16,16) → (224,224) and normalise to [0,1]."""
        big = cv2.resize(
            attn_16x16.astype(np.float32),
            (224, 224),
            interpolation=cv2.INTER_CUBIC,
        )
        mn, mx = big.min(), big.max()
        if mx > mn:
            big = (big - mn) / (mx - mn)
        return big


# ── Standalone self-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")   # change to Qt5Agg if needed

    print("dashboard_screen1.py — self test with dummy data")
    print("Close the window to exit.\n")

    fig, ax = plt.subplots(figsize=(5, 5.5), facecolor="#0F172A")
    fig.suptitle("Screen 1 Self-Test", color="white", fontsize=10)
    panel = AgentViewPanel(ax)

    # Dummy frame — random coloured image
    rng      = np.random.default_rng(42)
    frame    = Image.fromarray((rng.integers(80, 200, (224, 224, 3))).astype(np.uint8))
    attn_map = rng.random((16, 16)).astype(np.float32)

    # Dummy YOLOE boxes
    boxes = [
        ("door",   20,  30,  80, 150),
        ("sign",  130,  10, 200,  45),
    ]

    def animate(i):
        surprise = 0.1 + 0.3 * np.sin(i * 0.2)
        action   = ["MoveAhead", "RotateRight", "RotateLeft"][i % 3]
        ocr      = "Room 204" if i % 20 == 0 else ""
        attn_cur = rng.random((16, 16)).astype(np.float32)
        panel.render(
            frame_pil    = frame,
            attn_map_16x16 = attn_cur,
            action_str   = action,
            yolo_boxes   = boxes if surprise > 0.25 else None,
            ocr_text     = ocr,
            surprise     = surprise,
        )
        return []

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, animate, frames=60, interval=150, blit=False)

    plt.tight_layout()
    plt.show()
    print("Self-test complete.")