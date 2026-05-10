import os, json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter

maps_dir = "data/saved_attnlrp_maps"

# Load valid pairs
npy_ids = set(f.replace("attnlrp_","").replace(".npy","")
              for f in os.listdir(f"{maps_dir}/attnlrp"))
frame_ids = set(f.replace("frame_","").replace(".png","")
                for f in os.listdir(f"{maps_dir}/frames"))
valid_ids = sorted(npy_ids.intersection(frame_ids))

meta = json.load(open(f"{maps_dir}/metadata.json"))
meta_dict = {str(e["frame_id"]).zfill(3): e for e in meta}

frames, attnlrp_maps, actions, surprises = [], [], [], []
for fid in valid_ids:
    frames.append(
        Image.open(f"{maps_dir}/frames/frame_{fid}.png"))
    attnlrp_maps.append(
        np.load(f"{maps_dir}/attnlrp/attnlrp_{fid}.npy"))
    entry = meta_dict.get(fid, {})
    actions.append(entry.get("action", "Unknown"))
    surprises.append(entry.get("surprise", 0.0))

print(f"Loaded {len(frames)} valid samples")

# Cluster
X = np.stack([m.flatten() for m in attnlrp_maps])
X = normalize(X)
best_k, best_score, best_labels = 3, -1, None
for k in [3, 4, 5]:
    labels = KMeans(
        n_clusters=k, random_state=42, n_init=10
    ).fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k={k}: silhouette={score:.3f}")
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels

print(f"Best k={best_k}, silhouette={best_score:.3f}")

# Visualise
os.makedirs("cluster_visualisations", exist_ok=True)
for cluster_id in range(best_k):
    idxs = [i for i,l in enumerate(best_labels)
            if l==cluster_id][:6]
    fig, axes = plt.subplots(2, 6, figsize=(18,6))
    fig.patch.set_facecolor("#0F172A")
    for j, idx in enumerate(idxs):
        axes[0,j].imshow(frames[idx])
        axes[0,j].set_title(
            actions[idx], color="white", fontsize=8)
        axes[0,j].axis("off")
        axes[1,j].imshow(attnlrp_maps[idx], cmap="jet")
        axes[1,j].set_title(
            f"surprise={surprises[idx]:.2f}",
            color="white", fontsize=8)
        axes[1,j].axis("off")
    for j in range(len(idxs), 6):
        axes[0,j].axis("off")
        axes[1,j].axis("off")
    plt.suptitle(
        f"Cluster {cluster_id} — Saliency Attribution",
        color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"cluster_visualisations/cluster_{cluster_id}_attnlrp.png",
        facecolor="#0F172A", dpi=100)
    plt.close()
    print(f"Cluster {cluster_id} saved")