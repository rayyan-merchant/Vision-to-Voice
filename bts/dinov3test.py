import torch

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino.eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    f = dino.forward_features(x)

print("CLS shape:", f["x_norm_clstoken"].shape) 
print("Patch shape:", f["x_norm_patchtokens"].shape) 
print("DINOv3 distilled READY")
