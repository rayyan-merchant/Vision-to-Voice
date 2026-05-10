import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class AgentViewDashboard:
    def __init__(self, window_title="Vision-to-Voice | Agent View"):
        # Create figure with dark background
        self.fig, self.ax = plt.subplots(figsize=(6.4, 6.8), dpi=100)
        self.fig.canvas.manager.set_window_title(window_title)
        self.fig.patch.set_facecolor('#0F172A')
        
        # Panel Title
        self.ax.set_title("Agent View + DINOv2 Attention", color='white', fontsize=11, pad=10)
        
        # Remove ticks but keep the border
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#334155')
            spine.set_linewidth(2)
            
        # Remove margins inside the plot
        self.ax.margins(0, 0)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        
        # Initialize image and UI element caches for fast rendering
        self.im_frame = None
        self.im_attn = None
        self.dynamic_elements = []

    def update(self, frame_rgb, attn_map, action, surprise, step, total_steps, yoloe_detections=None, ocr_text=None, surprise_threshold=4.0):
        h, w = frame_rgb.shape[:2]
        scale_x = 640 / w
        scale_y = 640 / h
        
        # Upscale frame to 640x640 for visibility
        frame_disp = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # Handle tensor detachment if passed as PyTorch tensor
        if hasattr(attn_map, 'detach'):
            attn_map = attn_map.detach().cpu().numpy()
            
        # Resize attention map from (16,16) to (224,224) and normalize to 0-1
        if attn_map.shape != (224, 224):
            attn_map_224 = cv2.resize(attn_map.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
        else:
            attn_map_224 = attn_map.astype(np.float32)
            
        attn_min, attn_max = np.min(attn_map_224), np.max(attn_map_224)
        if attn_max - attn_min > 1e-8:
            attn_norm = (attn_map_224 - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = attn_map_224
            
        # Resize normalized map to fit the 640x640 upscaled display
        attn_disp = cv2.resize(attn_norm, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # Update image data efficiently
        if self.im_frame is None:
            self.im_frame = self.ax.imshow(frame_disp)
            self.im_attn = self.ax.imshow(attn_disp, cmap='jet', alpha=0.35)
        else:
            self.im_frame.set_data(frame_disp)
            self.im_attn.set_data(attn_disp)
            
        # Clear previous dynamic elements (bboxes, text, banners)
        for el in self.dynamic_elements:
            el.remove()
        self.dynamic_elements.clear()
        
        # Bounding Box colors per class
        colors = {
            'door': '#38BDF8',
            'sign': '#A3E635',
            'person': '#EF4444',
            'others': '#FB923C'
        }
        
        if yoloe_detections:
            for det in yoloe_detections:
                bbox = det.get('bbox', [0, 0, 0, 0])
                label = det.get('label', 'others')
                conf = det.get('conf', 0.0)
                color = colors.get(label.lower(), colors['others'])
                
                # Scale pixel coordinates
                x1, y1, x2, y2 = bbox
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                
                # Draw Box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
                self.ax.add_patch(rect)
                self.dynamic_elements.append(rect)
                
                # Draw Label
                text_str = f"{label} {conf:.2f}"
                t = self.ax.text(x1, y1 - 5, text_str, color='white', fontsize=9, fontweight='bold',
                                 bbox=dict(facecolor='#0F172A', edgecolor=color, alpha=0.8, pad=2))
                self.dynamic_elements.append(t)
                
        # Popup banner for EasyOCR
        if ocr_text:
            rect = patches.Rectangle((0, 0), 640, 40, linewidth=0, edgecolor='none', facecolor='black', alpha=0.6)
            self.ax.add_patch(rect)
            self.dynamic_elements.append(rect)
            
            t = self.ax.text(320, 20, f"Sign: {ocr_text}", color='white', va='center', ha='center', fontsize=11, fontweight='bold')
            self.dynamic_elements.append(t)
            
        # Status bar at the bottom
        sb_height = 40
        rect = patches.Rectangle((0, 640 - sb_height), 640, sb_height, linewidth=0, edgecolor='none', facecolor='#1E293B')
        self.ax.add_patch(rect)
        self.dynamic_elements.append(rect)
        
        # Action field
        t1 = self.ax.text(15, 640 - sb_height/2, f"Action: {action}", color='white', va='center', ha='left', fontsize=10, fontweight='bold')
        self.dynamic_elements.append(t1)
        
        # Surprise field
        try:
            surprise_val = float(surprise)
        except (ValueError, TypeError):
            surprise_val = 0.0
            
        surprise_color = '#EF4444' if surprise_val > surprise_threshold else 'white'
        t2 = self.ax.text(320, 640 - sb_height/2, f"Surprise: {surprise_val:.2f}", color=surprise_color, va='center', ha='center', fontsize=10, fontweight='bold')
        self.dynamic_elements.append(t2)
        
        # Step counter field
        try:
            step_val = int(step)
            total_val = int(total_steps)
        except (ValueError, TypeError):
            step_val = 0
            total_val = 0
            
        t3 = self.ax.text(625, 640 - sb_height/2, f"Step: {step_val:03d} / {total_val}", color='white', va='center', ha='right', fontsize=10, fontweight='bold')
        self.dynamic_elements.append(t3)

    def show(self):
        # Render without blocking
        plt.pause(0.05)
        
    def save_frame(self, path):
        # Save figure as PNG, maintaining exactly the same size to ensure uniform video frames
        self.fig.savefig(path, facecolor=self.fig.get_facecolor(), edgecolor='none')