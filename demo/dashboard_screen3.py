import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class CognitiveMapDashboard:
    def __init__(self, window_title="Vision-to-Voice | Cognitive Map"):
        self.fig = plt.figure(figsize=(8, 9), dpi=100)
        self.fig.canvas.manager.set_window_title(window_title)
        self.fig.patch.set_facecolor('#0F172A')
        
        # GridSpec with 2 rows, ratio 5:1
        self.gs = GridSpec(2, 1, height_ratios=[5, 1], figure=self.fig, hspace=0.05)
        
        self.ax_graph = self.fig.add_subplot(self.gs[0])
        self.ax_stats = self.fig.add_subplot(self.gs[1])
        
        self.cmap = plt.cm.RdYlBu_r
        
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        self._format_axes()

    def _format_axes(self):
        self.ax_graph.set_facecolor('#1E293B')
        self.ax_stats.set_facecolor('#1E293B')
        
        for ax in [self.ax_graph, self.ax_stats]:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor('#334155')
                spine.set_linewidth(2)
                
        self.ax_graph.set_title("Cognitive Map — Live Exploration", color='white', fontsize=11, pad=10)

    def update(self, graph, current_nid):
        self.ax_graph.clear()
        self.ax_stats.clear()
        self._format_axes()
        
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        if num_nodes == 0:
            self.ax_graph.text(0.5, 0.5, "Exploring...", color="white", fontsize=14, 
                               ha="center", va="center", transform=self.ax_graph.transAxes)
            self._update_stats(0, 0, 0.0)
            return
            
        pos = {}
        surp_vals = []
        labels = {}
        for n in graph.nodes:
            # Spatial mapping using literal AI2-THOR [x, z] layout coords
            node_pos = graph.nodes[n].get("pos", [0, 0])
            pos[n] = (node_pos[0], node_pos[1])
            
            surp_vals.append(graph.nodes[n].get("surprise", 0.0))
            if graph.nodes[n].get("label") is not None:
                labels[n] = graph.nodes[n]["label"]
                
        surp_vals = np.array(surp_vals)
        s_min = surp_vals.min()
        s_max = surp_vals.max()
        
        # Normalize between 0-1 for cmap, safely handle 0 deviation
        if s_max - s_min > 1e-8:
            norm_surp = (surp_vals - s_min) / (s_max - s_min)
        else:
            norm_surp = np.zeros_like(surp_vals)
            
        if num_edges > 0:
            nx.draw_networkx_edges(graph, pos, ax=self.ax_graph, edge_color="#475569", width=0.8)
            
        nx.draw_networkx_nodes(graph, pos, ax=self.ax_graph, 
                               node_color=norm_surp, cmap=self.cmap, 
                               node_size=80, vmin=0, vmax=1)
                               
        for n, label in labels.items():
            self.ax_graph.annotate(label, xy=pos[n], xytext=(0, -12), textcoords="offset points", 
                                   color="white", fontsize=7, ha="center")
                                   
        # Current position marker: base circular node and overlaid star
        if current_nid in pos:
            cur_pos = pos[current_nid]
            self.ax_graph.scatter([cur_pos[0]], [cur_pos[1]], s=200, c='#FDE047', edgecolors='white', zorder=4)
            self.ax_graph.scatter([cur_pos[0]], [cur_pos[1]], marker="*", s=400, c="#FDE047", zorder=5, edgecolors="white", linewidths=1.5)
            
        self.ax_graph.margins(0.15)
            
        # Draw colorbar using inset_axes
        cax = self.ax_graph.inset_axes([0.88, 0.65, 0.04, 0.3])
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, cax=cax, orientation='vertical')
        
        # Flip colorbar so blue=low is at the top, red=high is at the bottom
        cbar.ax.invert_yaxis()
        
        cbar.set_label("Surprise", color="white", fontsize=9)
        cbar.set_ticks([0.0, 1.0])
        max_str = f"{s_max:.2f}" if s_max > 0 else "0.00"
        cbar.set_ticklabels(['0.0', max_str])
        cbar.ax.tick_params(colors="white", labelsize=8)
        
        if hasattr(cbar.outline, 'set_edgecolor'):
            cbar.outline.set_edgecolor('#334155')
        
        avg_surp = surp_vals.mean() if len(surp_vals) > 0 else 0.0
        self._update_stats(num_nodes, num_edges, avg_surp)

    def _update_stats(self, nodes, edges, avg_surprise):
        self.ax_stats.text(0.1, 0.5, f"Nodes: {nodes}", color="white", fontsize=12, ha="left", va="center", transform=self.ax_stats.transAxes)
        self.ax_stats.text(0.5, 0.5, f"Edges: {edges}", color="white", fontsize=12, ha="center", va="center", transform=self.ax_stats.transAxes)
        self.ax_stats.text(0.9, 0.5, f"Avg Surprise: {avg_surprise:.3f}", color="white", fontsize=12, ha="right", va="center", transform=self.ax_stats.transAxes)

    def show(self):
        plt.pause(0.05)
        
    def save_frame(self, path):
        self.fig.savefig(path, facecolor=self.fig.get_facecolor(), edgecolor='none')